from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def gerar_texto(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    print("Aplicação de Geração de Texto com IA")
    while True:
        prompt = input("Digite o prompt (ou 'sair' para encerrar): ")
        if prompt.lower() == 'sair':
            break
        resposta = gerar_texto(prompt)
        print("Resposta da IA:")
        print(resposta)

if __name__ == "__main__":
    main()
