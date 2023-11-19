using System;
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

        public Task<string> RunAsync(string prompt, int outputLength = 32, IProgress<string> progress = null, CancellationToken? cancellationToken = null)
        {
            return Task.Run(() => {
                var tokenSource = cancellationToken != null ? CancellationTokenSource.CreateLinkedTokenSource(disposeCancellationTokenSource.Token, cancellationToken.Value) : disposeCancellationTokenSource;
                return Run(prompt, contextPointer, outputLength, progress, tokenSource.Token);
            });
        }

        string Run(string prompt, IntPtr ctx, int outputLength, IProgress<string> progress = null, CancellationToken? cancellationToken = null)
        {
            StringBuilder outputStringBuilder = new StringBuilder();

            int eosToken = EosToken;
            int[] tokens = TokenizePrompt(prompt, true);

            int totalTokens = tokens.Length + outputLength;
            if (totalTokens > ContextSize)
                throw new LlamaException($"Error: Model context size {ContextSize} tokens can't fit total of {totalTokens} tokens expected");

            LlamaLibrary.LlamaBatch batch = CreateBatch(tokens, totalTokens);

            int decodeResult = LlamaLibrary.llama_decode(ctx, batch);
            if (decodeResult != 0)
                throw new LlamaException($"llama_decode() failed Code: {decodeResult}");

            for (int i = batch.n_tokens; i < totalTokens; i++)
            {
                // Sample the next token
                LlamaLibrary.LlamaTokenData[] candidates = FindCandidates(ctx, batch);

                int newTokenId = SampleToken(ctx, candidates);
                    
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
                if (LlamaLibrary.llama_decode(ctx, batch) != 0)
                    throw new LlamaException("llama_decode() failed");
            }
            return outputStringBuilder.ToString();
        }

        public int GetVocabLength()
        {
            return LlamaLibrary.llama_n_vocab(modelPointer);
        }

        static unsafe int SampleToken(IntPtr ctx, LlamaLibrary.LlamaTokenData[] candidates)
        {
            fixed (LlamaLibrary.LlamaTokenData* pCandidates = candidates)
            {
                var candidatesArray = new LlamaLibrary.LlamaTokenDataArray
                {
                    data = new IntPtr(pCandidates),
                    size = candidates.Length,
                    sorted = false
                };

                // Sample the most likely token
                int newTokenId = LlamaLibrary.llama_sample_token_greedy(ctx, ref candidatesArray);
                return newTokenId;
            }
        }

        public unsafe LlamaLibrary.LlamaTokenData[] FindCandidates(IntPtr ctx, LlamaLibrary.LlamaBatch batch)
        {
            IntPtr logitsPtr = LlamaLibrary.llama_get_logits_ith(ctx, batch.n_tokens - 1);
            int vocabLength = GetVocabLength();
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
    }
}