using System;
using System.IO;
using UnityEngine;

namespace Abuksigun.LlamaCpp
{
    [ExecuteInEditMode]
    public class LlamaExample : MonoBehaviour
    {
        LlamaModel model;

        // Download model here: https://huggingface.co/TheBloke/speechless-mistral-dolphin-orca-platypus-samantha-7B-GGUF/blob/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf
        [SerializeField] string modelPath = "StreamingAssets/Models/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf";
        [SerializeField] string systemPrompt = "You are an AI game character";
        [SerializeField] string userPrompt = "You are in a Tavern\nHP:40%\nWhat is your next action:";
        [SerializeField] string assistantPrompt = "I will";

        [ContextMenu("Run")]
        public async void RunAsync()
        {
            const string promptFormat = "<|im_start|>system\n{{system}}\n<|im_end|>\n<|im_start|>user\n{{user}}\n<|im_end|>\n<|im_start|>assistant\n{{assistant}}";

            string fullModelPath = Path.Join(Application.streamingAssetsPath, modelPath);
            model ??= await LlamaModel.LoadModel(fullModelPath, new Progress<float>(x => Debug.Log($"Progress {x}")));
            if (model == null)
            {
                Debug.LogError("Failed to load model");
                return;
            }
            string result = await model.RunAsync(FormatPrompt(promptFormat, systemPrompt, userPrompt, assistantPrompt), 100, new Progress<string>(x => Debug.Log(x)));
            Debug.Log($"Result: {result}");
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