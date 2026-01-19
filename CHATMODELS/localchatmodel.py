from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set cache directory
os.environ["HF_HOME"] = "D:/huggingface_cache"

# Initialize the pipeline
# device=-1 uses CPU, device=0 uses GPU if available
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100, 
        "temperature": 0.5,
        "do_sample": True
    },
    device=-1  # Change to 0 if you have installed the GPU version of PyTorch
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    print(f"An error occurred: {e}")