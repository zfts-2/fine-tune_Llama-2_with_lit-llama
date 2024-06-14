import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def main():
    device = torch.device("cuda:3")

    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("Llama-2-7b-hf/tokenizer.model")

    # Load the model
    model = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf/", torch_dtype=torch.float16).to(device)

    model.eval()

    # Sample input text
    input_text = """
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
"""
    model_input = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output with sampling
    with torch.no_grad():
        outputs = model.generate(
            **model_input,
            max_length=256,
            do_sample=True,
            top_p=0.6
        )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()