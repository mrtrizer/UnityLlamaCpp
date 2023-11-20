using System;
using System.IO;
using System.Threading;
using UnityEngine;

namespace Abuksigun.LlamaCpp
{
    [ExecuteInEditMode]
    public class LlamaExample : MonoBehaviour
    {
        CancellationTokenSource cts;
        LlamaModel model;

        // Download model here: https://huggingface.co/TheBloke/speechless-mistral-dolphin-orca-platypus-samantha-7B-GGUF/blob/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf
        [SerializeField] string modelPath = "StreamingAssets/Models/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf";
        [SerializeField, TextArea(10, 10)] string systemPrompt = "You are an AI game character";
        [SerializeField, TextArea(10, 10)] string userPrompt = "You are in a Tavern\nHP:40%\nWhat is your next action:";
        [SerializeField, TextArea(10, 10)] string assistantPrompt = "I will";

        [SerializeField, Range(0, 1.5f)] float temperature = 0.8f;

        [ContextMenu("Run")]
        public async void RunAsync()
        {
            const string promptFormat = "<|im_start|>system\n{{system}}\n<|im_end|>\n<|im_start|>user\n{{user}}\n<|im_end|>\n<|im_start|>assistant\n{{assistant}}";
            const string customEos = "<|im_end|>";

            string fullModelPath = Path.Join(Application.streamingAssetsPath, modelPath);
            model ??= await LlamaModel.LoadModel(fullModelPath, new Progress<float>(x => Debug.Log($"Progress {x}")));
            Debug.Log($"Model context size: {model.ContextSize} tokens.");

            cts = new CancellationTokenSource();
            void Progress(string currentString)
            {
                if (currentString.EndsWith(customEos))
                {
                    cts.Cancel();
                    cts = null;
                }
                Debug.Log(currentString);
            }
            string fullPrompt = FormatPrompt(promptFormat, systemPrompt, userPrompt, assistantPrompt);
            Debug.Log(fullPrompt);
            var parameters = new LlamaModel.SamplingParams { Temp = temperature };
            string result = await model.RunAsync(fullPrompt, 100, parameters, new Progress<string>(Progress), cts.Token);
            Debug.Log($"Result: {result}");
        }

        [ContextMenu("Stop")]
        public void Stop()
        {
            cts?.Cancel();
        }

        public static string FormatPrompt(string promptFormat, string system, string user, string assistant = "")
        {
            return promptFormat
                .Replace("{{system}}", system)
                .Replace("{{user}}", user)
                .Replace("{{assistant}}", assistant);
        }
    }
}