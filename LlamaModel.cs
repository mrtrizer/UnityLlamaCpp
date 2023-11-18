using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Abuksigun.LlamaCpp
{
    public sealed class LlamaModel : IDisposable
    {
        IntPtr nativePointer;
        readonly CancellationTokenSource disposeCancellationTokenSource = new();

        public IntPtr NativePointer => nativePointer;
        public int EosToken => LlamaLibrary.llama_token_eos(nativePointer);


        public static async Task<LlamaModel> LoadModel(string modelPath, IProgress<float> progress)
        {
            IntPtr modelPtr = await Task.Run<IntPtr>(() =>
            {
                LlamaLibrary.llama_backend_init(numa: false);
                void MyProgressCallback(float progressFloat, IntPtr userData) => progress.Report(progressFloat);
                var modelParams = new LlamaLibrary.LlamaModelParams(MyProgressCallback, IntPtr.Zero);
                return LlamaLibrary.llama_load_model_from_file(modelPath, modelParams);
            });
            return modelPtr != IntPtr.Zero ? new LlamaModel(modelPtr) : null;
        }

        LlamaModel(IntPtr modelPtr)
        {
            nativePointer = modelPtr;
        }

        ~LlamaModel()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (nativePointer != IntPtr.Zero)
            {
                disposeCancellationTokenSource.Cancel();
                LlamaLibrary.llama_free_model(nativePointer);
                LlamaLibrary.llama_backend_free();
                nativePointer = IntPtr.Zero;
            }
        }

        public Task<string> RunAsync(string prompt, int outputLength = 32, IProgress<string> progress = null, CancellationToken? cancellationToken = null)
        {
            int threadsN = SystemInfo.processorCount;
            return Task.Run(() => {
                var ctxParams = new LlamaLibrary.LlamaContextParams(1234, (uint)threadsN);
                ctxParams.n_ctx = 2048;
                IntPtr ctx = CreateContext(ctxParams);
                if (ctx == IntPtr.Zero)
                    throw new Exception("Failed to create the llama_context");

                try
                {
                    var tokenSource = cancellationToken != null ? CancellationTokenSource.CreateLinkedTokenSource(disposeCancellationTokenSource.Token, cancellationToken.Value) : disposeCancellationTokenSource;
                    return Run(prompt, ctx, outputLength, progress, tokenSource.Token);
                }
                finally
                {
                    LlamaLibrary.llama_free(ctx);
                }
            });
        }

        string Run(string prompt, IntPtr ctx, int outputLength, IProgress<string> progress = null, CancellationToken? cancellationToken = null)
        {
            StringBuilder outputStringBuilder = new StringBuilder();

            int[] tokens = TokenizePrompt(prompt, true);

            int nCtx = LlamaLibrary.llama_n_ctx(ctx); // Get the context count
            int nKvReq = tokens.Length + (outputLength - tokens.Length);

            Debug.Log($"n_len = {outputLength}, n_ctx = {nCtx}, n_kv_req = {nKvReq}");

            if (nKvReq > nCtx)
                throw new Exception("Error: n_kv_req > n_ctx, the required KV cache size is not big enough");

            // This object is used to submit token data for decoding
            LlamaLibrary.LlamaBatch batch = CreateBatch(tokens);

            if (LlamaLibrary.llama_decode(ctx, batch) != 0)
                throw new Exception("llama_decode() failed");

            for (int i = batch.n_tokens; i < outputLength; i++)
            {
                // Sample the next token
                LlamaLibrary.LlamaTokenData[] candidates = FindCandidates(ctx, batch);

                int newTokenId = SampleToken(ctx, candidates);
                    
                if (newTokenId == EosToken || i == outputLength)
                    break;

                // Output the generated text
                string tokenText = LlamaTokenToPiece(newTokenId);
                outputStringBuilder.Append(tokenText);
                progress?.Report(outputStringBuilder.ToString());
                batch.n_tokens = 0;

                // push this new token for next evaluation
                LlamaBatchAdd(ref batch, newTokenId, i, new int[] { 0 }, true);

                if (cancellationToken?.IsCancellationRequested ?? false)
                    break;
                if (LlamaLibrary.llama_decode(ctx, batch) != 0)
                    throw new Exception("llama_decode() failed");
            }
            return outputStringBuilder.ToString();
        }

        public int GetVocabLength()
        {
            return LlamaLibrary.llama_n_vocab(nativePointer);
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
            // Cast IntPtr to a float pointer
            float* logits = (float*)logitsPtr.ToPointer();

            for (int j = 0; j < vocabLength; j++)
                candidates[j] = new LlamaLibrary.LlamaTokenData { id = j, logit = logits[j], p = 0.0f };
            return candidates;
        }

        public static LlamaLibrary.LlamaBatch CreateBatch(int[] tokens)
        {
            LlamaLibrary.LlamaBatch batch = LlamaLibrary.llama_batch_init(512, 0, 1);

            for (int i = 0; i < tokens.Length; i++)
                LlamaBatchAdd(ref batch, tokens[i], i, new int[] { 0 }, false);

            unsafe
            {
                // FIXME: Is this really needed?
                // Ensure logits are output for the last token of the prompt
                batch.logits[batch.n_tokens - 1] = 1;
            }

            return batch;
        }

        public unsafe static void LlamaBatchAdd(ref LlamaLibrary.LlamaBatch batch, int id, int pos, int[] seqIds, bool logits)
        {
            // Ensure batch has enough space to add a new token
            // This check depends on how batch is initialized and managed

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
            int nTokens = LlamaLibrary.llama_tokenize(nativePointer, prompt, prompt.Length, tokens, tokens.Length, addBos, false);
            Array.Resize(ref tokens, nTokens);
            return tokens;
        }

        public unsafe string LlamaTokenToPiece(int token)
        {
            const int initialSize = 16;
            byte[] buffer = new byte[initialSize];
            fixed (byte* pBuffer = buffer)
            {
                int nTokens = LlamaLibrary.llama_token_to_piece(nativePointer, token, pBuffer, buffer.Length);
                if (nTokens < 0)
                {
                    Array.Resize(ref buffer, -nTokens);
                    int check = LlamaLibrary.llama_token_to_piece(nativePointer, token, pBuffer, buffer.Length);
                    if (check == -nTokens)
                        return null;
                }
                else
                {
                    Array.Resize(ref buffer, nTokens);
                }
            }

            string result = Encoding.UTF8.GetString(buffer);
            return result;
        }

        internal IntPtr CreateContext(LlamaLibrary.LlamaContextParams ctxParams)
        {
            return LlamaLibrary.llama_new_context_with_model(nativePointer, ctxParams);
        }
    }
}