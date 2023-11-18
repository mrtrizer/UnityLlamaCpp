# UnityLlamaCpp
Connect llama.cpp to Unity3d in two clicks

# Installation:
- Add git repo as package Window -> Package Manager -> Add from Git URL https://github.com/mrtrizer/UnityLlamaCpp.git
- Download a GGUF model, for example this - https://huggingface.co/TheBloke/speechless-mistral-dolphin-orca-platypus-samantha-7B-GGUF/blob/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf
- Put model file in StreamingAssets/Models
- Find Test.prefab in package dir and use component context menu to Run it, it should generate some response to a prompt
- Use LlamaExample.cs as and example
