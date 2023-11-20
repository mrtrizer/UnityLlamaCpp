using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Abuksigun.LlamaCpp
{
    public sealed class LlamaModel : IDisposable
    {
        public class LlamaException : Exception
        {
            public LlamaException(string message) : base(message) { }
        }

        IntPtr modelPointer;
        IntPtr contextPointer;
        readonly CancellationTokenSource disposeCancellationTokenSource = new();

        public IntPtr NativeModelPointer => modelPointer;
        public IntPtr NativeContextPointer => contextPointer;
        public int EosToken => LlamaLibrary.llama_token_eos(modelPointer);
        public int ContextSize => LlamaLibrary.llama_n_ctx(contextPointer);
        public int VocabLength => LlamaLibrary.llama_n_vocab(modelPointer);

        public static async Task<LlamaModel> LoadModel(string modelPath, IProgress<float> progress)
        {
            int threadsN = SystemInfo.processorCount;
            (IntPtr newModelPointer, IntPtr newContextPointer) = await Task.Run<(IntPtr, IntPtr)>(() =>
            {
                LlamaLibrary.llama_backend_init(numa: false);

                var modelParams = new LlamaLibrary.LlamaModelParams((float progressFloat, IntPtr _) => progress.Report(progressFloat), IntPtr.Zero);
                try
                {
                    IntPtr model = LlamaLibrary.llama_load_model_from_file(modelPath, modelParams);
                    if (model == IntPtr.Zero)
                        throw new LlamaException("Failed to load the Llama model");
                    try
                    {
                        var ctxParams = new LlamaLibrary.LlamaContextParams(1234, (uint)threadsN);
                        IntPtr ctx = LlamaLibrary.llama_new_context_with_model(model, ctxParams);
                        if (ctx == IntPtr.Zero)
                            throw new LlamaException("Failed to create the Llama context");
                        return (model, ctx);
                    }
                    catch
                    {
                        LlamaLibrary.llama_free_model(model);
                        throw;
                    }
                }
                catch
                {
                    LlamaLibrary.llama_backend_free();
                    throw;
                }
            });
            return new LlamaModel(newModelPointer, newContextPointer);
        }

        LlamaModel(IntPtr modelPointer, IntPtr contextPointer)
        {
            this.modelPointer = modelPointer;
            this.contextPointer = contextPointer;
        }

        ~LlamaModel()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (modelPointer == IntPtr.Zero && contextPointer == IntPtr.Zero)
                return;
            disposeCancellationTokenSource.Cancel();
            if (contextPointer != IntPtr.Zero)
            {
                LlamaLibrary.llama_free(contextPointer);
                contextPointer = IntPtr.Zero;
            }
            if (modelPointer != IntPtr.Zero)
            {
                LlamaLibrary.llama_free_model(modelPointer);
                LlamaLibrary.llama_backend_free();
                modelPointer = IntPtr.Zero;
            }
        }

        public Task<string> RunAsync(string prompt, int outputLength = 32, SamplingParams samplingParams = null, IProgress<string> progress = null, CancellationToken? ct = null)
        {
            return Task.Run(() => {
                var tokenSource = ct != null ? CancellationTokenSource.CreateLinkedTokenSource(disposeCancellationTokenSource.Token, ct.Value) : disposeCancellationTokenSource;
                return Run(prompt, contextPointer, outputLength, samplingParams ?? new(), progress, tokenSource.Token);
            });
        }

        string Run(string prompt, IntPtr context, int outputLength, SamplingParams samplingParams, IProgress<string> progress = null, CancellationToken? cancellationToken = null)
        {
            StringBuilder outputStringBuilder = new StringBuilder();

            int eosToken = EosToken;
            int[] tokens = TokenizePrompt(prompt, true);
            
            var samplingContext = new LlamaSamplingContext(samplingParams, tokens);
            
            int totalTokens = tokens.Length + outputLength;
            if (totalTokens > ContextSize)
                throw new LlamaException($"Error: Model context size {ContextSize} tokens can't fit total of {totalTokens} tokens expected");

            LlamaLibrary.LlamaBatch batch = CreateBatch(tokens, totalTokens);

            int decodeResult = LlamaLibrary.llama_decode(context, batch);
            if (decodeResult != 0)
                throw new LlamaException($"llama_decode() failed Code: {decodeResult}");

            for (int i = batch.n_tokens; i < totalTokens; i++)
            {
                int newTokenId = SampleToken(samplingContext, batch.n_tokens - 1);
                
                samplingContext.AddToken(newTokenId);
                
                if (newTokenId == eosToken)
                    break;

                // Output the generated text
                string tokenText = LlamaTokenToPiece(newTokenId);
                outputStringBuilder.Append(tokenText);
                progress?.Report(outputStringBuilder.ToString());
                batch.n_tokens = 0;

                // push this new token for next evaluation
                LlamaBatchAdd(ref batch, newTokenId, i, true, 0);

                if (cancellationToken?.IsCancellationRequested ?? false)
                    break;
                if (LlamaLibrary.llama_decode(context, batch) != 0)
                    throw new LlamaException("llama_decode() failed");
            }
            return outputStringBuilder.ToString();
        }

        unsafe int SampleTokenGreedy(IntPtr ctx, int idx)
        {
            LlamaLibrary.LlamaTokenData[] candidates = FindCandidates(ctx, idx);

            fixed (LlamaLibrary.LlamaTokenData* pCandidates = candidates)
            {
                var candidatesArray = new LlamaLibrary.LlamaTokenDataArray
                {
                    data = pCandidates,
                    size = candidates.Length,
                    sorted = false
                };

                // Sample the most likely token
                int newTokenId = LlamaLibrary.llama_sample_token_greedy(ctx, ref candidatesArray);
                return newTokenId;
            }
        }

        public unsafe LlamaLibrary.LlamaTokenData[] FindCandidates(IntPtr ctx, int idx)
        {
            IntPtr logitsPtr = LlamaLibrary.llama_get_logits_ith(ctx, idx);
            int vocabLength = VocabLength;
            LlamaLibrary.LlamaTokenData[] candidates = new LlamaLibrary.LlamaTokenData[vocabLength];
            
            float* logits = (float*)logitsPtr.ToPointer();
            for (int j = 0; j < vocabLength; j++)
                candidates[j] = new LlamaLibrary.LlamaTokenData { id = j, logit = logits[j], p = 0.0f };
            return candidates;
        }

        public static LlamaLibrary.LlamaBatch CreateBatch(int[] tokens, int size)
        {
            LlamaLibrary.LlamaBatch batch = LlamaLibrary.llama_batch_init(size, 0, 1);

            for (int i = 0; i < tokens.Length; i++)
                LlamaBatchAdd(ref batch, tokens[i], i, false, 0);

            unsafe
            {
                // Ensure logits are output for the last token of the prompt
                batch.logits[batch.n_tokens - 1] = 1;
            }

            return batch;
        }

        public unsafe static void LlamaBatchAdd(ref LlamaLibrary.LlamaBatch batch, int id, int pos, bool logits, params int[] seqIds)
        {
            batch.token[batch.n_tokens] = id;
            batch.pos[batch.n_tokens] = pos;
            batch.n_seq_id[batch.n_tokens] = seqIds.Length;

            for (int i = 0; i < seqIds.Length; ++i)
            {
                batch.seq_id[batch.n_tokens][i] = seqIds[i];
            }

            batch.logits[batch.n_tokens] = logits ? (byte)1 : (byte)0;
            batch.n_tokens++;
        }

        public int[] TokenizePrompt(string prompt, bool addBos)
        {
            int[] tokens = new int[prompt.Length + (addBos ? 1 : 0)];
            int nTokens = LlamaLibrary.llama_tokenize(modelPointer, prompt, prompt.Length, tokens, tokens.Length, addBos, false);
            Array.Resize(ref tokens, nTokens);
            return tokens;
        }

        public string LlamaTokenToPiece(int token)
        {
            const int initialSize = 16;
            byte[] buffer = new byte[initialSize];

            int nTokens = LlamaLibrary.llama_token_to_piece(modelPointer, token, buffer, buffer.Length);
            if (nTokens < 0)
            {
                Array.Resize(ref buffer, -nTokens);
                int check = LlamaLibrary.llama_token_to_piece(modelPointer, token, buffer, buffer.Length);
                if (check == -nTokens)
                    return null;
            }
            else
            {
                Array.Resize(ref buffer, nTokens);
            }

            string result = Encoding.UTF8.GetString(buffer);
            return result;
        }

        public unsafe class LlamaSamplingContext
        {
            public SamplingParams Params { get; }
            public int[] Prev { get; }
            public List<LlamaLibrary.LlamaTokenData> Cur { get; } = new();
            public LlamaLibrary.LlamaGrammar* Grammar { get; }

            public LlamaSamplingContext(SamplingParams parameters, int[] promptTokens)
            {
                Params = parameters;
                int fillLength = Mathf.Max(parameters.NPrev - promptTokens.Length, 0);
                int skipLength = Mathf.Max(promptTokens.Length - parameters.NPrev, 0);
                Prev = Enumerable.Repeat(0, fillLength).Concat(promptTokens.Skip(skipLength)).ToArray();
            }

            public void AddToken(int id)
            {
                for (int i = 0; i < Prev.Length - 1; i++)
                    Prev[i] = Prev[i + 1];
                Prev[Prev.Length - 1] = id;
            }
        }

        public class SamplingParams
        {
            public float Temp { get; set; } = 0.80f;
            public int TopK { get; set; } = 40;
            public float TopP { get; set; } = 0.95f;
            public float MinP { get; set; } = 0.05f;
            public float TfsZ { get; set; } = 1.00f;
            public float TypicalP { get; set; } = 1.00f;
            public int PenaltyLastN { get; set; } = 64;
            public float PenaltyRepeat { get; set; } = 1.10f;
            public float PenaltyFreq { get; set; } = 0.00f;
            public float PenaltyPresent { get; set; } = 0.00f;
            public bool PenalizeNl { get; set; } = true;
            public Dictionary<int, float> LogitBias { get; set; } = new Dictionary<int, float>();
            public int NPrev { get; set; } = 64;
            public int NProbs { get; set; } = 0;
        }

        public unsafe int SampleToken(LlamaSamplingContext samplingContext, int idx)
        {
            SamplingParams parameters = samplingContext.Params;

            int vocabLength = VocabLength;

            float temp = parameters.Temp;
            int topK = parameters.TopK <= 0 ? vocabLength : parameters.TopK;
            float topP = parameters.TopP;
            float minP = parameters.MinP;
            float tfsZ = parameters.TfsZ;
            float typicalP = parameters.TypicalP;
            int penaltyLastN = parameters.PenaltyLastN < 0 ? parameters.NPrev : parameters.PenaltyLastN;
            float penaltyRepeat = parameters.PenaltyRepeat;
            float penaltyFreq = parameters.PenaltyFreq;
            float penaltyPresent = parameters.PenaltyPresent;
            bool penalizeNl = parameters.PenalizeNl;

            var prev = samplingContext.Prev;
            var cur = samplingContext.Cur;

            IntPtr logitsPtr = LlamaLibrary.llama_get_logits_ith(contextPointer, idx);
            float[] logits = new float[vocabLength];
            Marshal.Copy(logitsPtr, logits, 0, vocabLength);

            foreach (var bias in parameters.LogitBias)
                logits[bias.Key] += bias.Value;

            cur.Clear();

            for (int tokenID = 0; tokenID < vocabLength; tokenID++)
                cur.Add(new LlamaLibrary.LlamaTokenData { id = tokenID, logit = logits[tokenID], p = 0 });

            var curArray = cur.ToArray();
            fixed (LlamaLibrary.LlamaTokenData* pCurArray = curArray)
            {
                LlamaLibrary.LlamaTokenDataArray curP = new LlamaLibrary.LlamaTokenDataArray
                {
                    data = pCurArray,
                    size = cur.Count,
                    sorted = false
                };

                if (prev.Length > 0)
                {
                    int nlTokenId = LlamaLibrary.llama_token_nl(modelPointer);
                    float nlLogit = logits[nlTokenId];

                    LlamaLibrary.llama_sample_repetition_penalties(contextPointer, &curP, prev, prev.Length, penaltyRepeat, penaltyFreq, penaltyPresent);

                    // If not penalizing new lines, reset the logit for the newline token
                    if (!penalizeNl)
                    {
                        for (int i = 0; i < curP.size; i++)
                        {
                            if (curP.data[i].id == nlTokenId)
                            {
                                curP.data[i].logit = nlLogit;
                                break;
                            }
                        }
                    }
                }

                int id = 0;
                if (temp < 0.0f)
                {
                    LlamaLibrary.llama_sample_softmax(contextPointer, &curP);
                    id = curP.data[0].id;
                }
                else if (temp == 0.0f)
                {
                    id = LlamaLibrary.llama_sample_token_greedy(contextPointer, ref curP);
                }
                else
                {
                    int minKeep = Math.Max(1, parameters.NProbs);

                    LlamaLibrary.llama_sample_top_k(contextPointer, &curP, topK, minKeep);
                    LlamaLibrary.llama_sample_tail_free(contextPointer, &curP, tfsZ, minKeep);
                    LlamaLibrary.llama_sample_typical(contextPointer, &curP, typicalP, minKeep);
                    LlamaLibrary.llama_sample_top_p(contextPointer, &curP, topP, minKeep);
                    LlamaLibrary.llama_sample_min_p(contextPointer, &curP, minP, minKeep);
                    LlamaLibrary.llama_sample_temp(contextPointer, &curP, temp);

                    id = LlamaLibrary.llama_sample_token(contextPointer, &curP);
                }

                return id;
            }
        }
    }
}