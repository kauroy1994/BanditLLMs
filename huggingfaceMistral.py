from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "I want to buy some chocolate, should I go to the shop or should I not? please return true or false"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print (tokenizer.batch_decode(generated_ids)[0])