# Semi-abandoned
Sadly, I don't have time to support this package at this moment, I recommend using https://github.com/SciSharp/LLamaSharp that supports the latest version of llama.cpp

# UnityLlamaCpp
Connect llama.cpp to Unity3d in two clicks

# MacOS/Linux/Windows CUDA
The bindings were made for b1518 version of llama.cpp, so if you need libraries for your OS, build from sources from release b1518 
You can find Windows CUDA llama.cpp dll here - [https://github.com/SciSharp/LLamaSharp](https://github.com/ggerganov/llama.cpp/releases/tag/b1518)

# Installation:
- Add git repo as package Window -> Package Manager -> Add from Git URL https://github.com/mrtrizer/UnityLlamaCpp.git
- Download a GGUF model, for example this - https://huggingface.co/TheBloke/speechless-mistral-dolphin-orca-platypus-samantha-7B-GGUF/blob/main/speechless-mistral-dolphin-orca-platypus-samantha-7b.Q4_K_M.gguf
- Put model file in StreamingAssets/Models
- Find Test.prefab in package dir and use component context menu to Run it, it should generate some response to a prompt
- Use LlamaExample.cs as and example
