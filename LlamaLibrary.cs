using System;
using System.Runtime.InteropServices;

namespace Abuksigun.LlamaCpp
{
    public unsafe static class LlamaLibrary
    {
        private const string DllName = "llama";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LlamaProgressCallback(float progress, IntPtr ctx);

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaModelParams
        {
            public int n_gpu_layers;
            public int main_gpu;
            public IntPtr tensor_split;
            public LlamaProgressCallback progress_callback;
            public IntPtr progress_callback_user_data;
            [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
            [MarshalAs(UnmanagedType.I1)] public bool use_mmap;
            [MarshalAs(UnmanagedType.I1)] public bool use_mlock;

            public LlamaModelParams(LlamaProgressCallback progressCallback, IntPtr progressCallbackUserData, int nGpuLayers = 0)
            {
                n_gpu_layers = nGpuLayers;
                main_gpu = 0;
                tensor_split = IntPtr.Zero;
                progress_callback = progressCallback;
                progress_callback_user_data = IntPtr.Zero;
                vocab_only = false;
                use_mmap = true;
                use_mlock = false;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaContextParams
        {
            public uint seed;
            public uint n_ctx;
            public uint n_batch;
            public uint n_threads;
            public uint n_threads_batch;
            public sbyte rope_scaling_type;
            public float rope_freq_base;
            public float rope_freq_scale;
            public float yarn_ext_factor;
            public float yarn_attn_factor;
            public float yarn_beta_fast;
            public float yarn_beta_slow;
            public uint yarn_orig_ctx;
            [MarshalAs(UnmanagedType.I1)] public bool mul_mat_q;
            [MarshalAs(UnmanagedType.I1)] public bool f16_kv;
            [MarshalAs(UnmanagedType.I1)] public bool logits_all;
            [MarshalAs(UnmanagedType.I1)] public bool embedding;

            public LlamaContextParams(uint seed, uint nThreads = 1, uint contextSize = 2048, sbyte ropeScaling = -1 )
            {
                this.seed = seed;
                n_ctx = contextSize;
                n_batch = contextSize;
                n_threads = nThreads;
                n_threads_batch = nThreads;
                rope_scaling_type = ropeScaling;
                rope_freq_base = 0.0f;
                rope_freq_scale = 0.0f;
                yarn_ext_factor = -1.0f;
                yarn_attn_factor = 1.0f;
                yarn_beta_fast = 32.0f;
                yarn_beta_slow = 1.0f;
                yarn_orig_ctx = 0;
                mul_mat_q = true;
                f16_kv = true;
                logits_all = false;
                embedding = false;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaTokenDataArray
        {
            public LlamaTokenData* data;
            public int size;
            [MarshalAs(UnmanagedType.I1)] public bool sorted;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaTokenData
        {
            public int id;
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        public unsafe struct LlamaBatch
        {
            public int n_tokens;
            public int* token;
            public float* embd;
            public int* pos;
            public int* n_seq_id;
            public int** seq_id;
            public byte* logits;

            // Legacy, may require removal in future llama.cpp versions
            private int _all_pos_0;
            private int _all_pos_1;
            private int _all_seq_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaGrammar
        {
            // const std::vector<std::vector<llama_grammar_element>> rules;
            // std::vector<std::vector<const llama_grammar_element*>> stacks;

            // llama_partial_utf8 partial_utf8;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_init(bool numa);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_load_model_from_file(string path_model, LlamaModelParams model_params);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free_model(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_ctx(IntPtr ctx);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern LlamaBatch llama_batch_init(int n_tokens, int embd, int n_seq_max);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_decode(IntPtr ctx, LlamaBatch batch);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_new_context_with_model(IntPtr model, LlamaContextParams ctx_params);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(IntPtr ctx);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_tokenize(IntPtr model, string text, int text_len, [MarshalAs(UnmanagedType.LPArray)] int[] tokens, int n_max_tokens, bool add_bos, bool special);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_logits(IntPtr ctx);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_get_logits_ith(IntPtr ctx, int i);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_n_vocab(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_sample_token_greedy(IntPtr ctx, ref LlamaTokenDataArray candidates);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_to_piece(IntPtr model, int token, [MarshalAs(UnmanagedType.LPArray)] byte[] buffer, int length);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_free();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_eos(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_nl(IntPtr model);



        // Sampling
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_repetition_penalties(IntPtr ctx, LlamaTokenDataArray* candidates, [MarshalAs(UnmanagedType.LPArray)] int[] lastTokens, int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_classifier_free_guidance(IntPtr ctx, LlamaTokenDataArray* candidates, IntPtr guidanceCtx, float scale);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_softmax(IntPtr ctx, LlamaTokenDataArray* candidates);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_top_k(IntPtr ctx, LlamaTokenDataArray* candidates, int k, int minKeep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_top_p(IntPtr ctx, LlamaTokenDataArray* candidates, float p, int minKeep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_min_p(IntPtr ctx, LlamaTokenDataArray* candidates, float p, int minKeep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_tail_free(IntPtr ctx, LlamaTokenDataArray* candidates, float z, int minKeep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_typical(IntPtr ctx, LlamaTokenDataArray* candidates, float p, int minKeep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_temp(IntPtr ctx, LlamaTokenDataArray* candidates, float temp);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_temperature(IntPtr ctx,  LlamaTokenDataArray* candidates, float temp);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sample_grammar(IntPtr ctx,  LlamaTokenDataArray* candidates, IntPtr grammar);
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_sample_token(IntPtr ctx, LlamaTokenDataArray* candidates);
    }
}