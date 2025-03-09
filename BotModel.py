# Import
import os, logging, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import test
if __name__ == "__main__":
    from Debug.Logging import loggingConfig
    loggingConfig()

# Đường dẫn model
model_path = "d:/AI_Model/GPT2_Model"
model_name = "gpt2"

# Check model
def check_model():
	if not os.path.exists(model_path):
		logging.warning("Không có model, đang tải...")
  
		os.makedirs(model_path, exist_ok=True)
		model = AutoModelForCausalLM.from_pretrained(model_name)
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model.save_pretrained(model_path)
		tokenizer.save_pretrained(model_path)
		
		logging.info(f"Model đã được tải và lưu vào: {model_path}")

	else:
		logging.info(f"Model đã có trong: {model_path}")
  
logging.info("Đã kiểm tra model!")

# Training model
def training_model():
	# Load model và tokenizer
	model = AutoModelForCausalLM.from_pretrained(model_path)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
    
	# Train data
	def train_data():
		logging.info("Bắt đầu train data...")
		
		# Load Data hội thoại
		data_path = "Memory/Data_training.json"
		with open(data_path, "r", encoding="utf-8") as f:
			dataset = json.load(f)
   
		# Kiểu dataset
		formatted_data = [f"{item['input']} {item['response']}" for item in dataset]

		# Data tokenizer
		inputs = tokenizer(formatted_data, padding=True, truncation=True, return_tensors="pt")
		labels = inputs.input_ids.clone()
		
		# Huấn luyện đơn giản
		optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
		model.train()
		
		# Luyện data
		for epoch in range(3):
			outputs = model(**inputs, labels = labels)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			logging.info(f"Epoch {epoch+1}: Loss = {loss.item()}")

		logging.info("Huấn luyện dữ liệu hội thoại hoàn tất!")
		model.save_pretrained(model_path)
	
  	# Train teencode
	def train_teencode():
		logging.info("Bắt đầu train data teen code...")
	
		# Data teen code
		slang_data_path = "Memory/Data_teencode.json"
		with open(slang_data_path, "r", encoding="utf-8") as f:
			dataset = json.load(f)
	
		# Kiểu dataset
		formatted_teencode = [f"{key} {value}" for key, value in dataset.items()]
	
		# Data tokenizer
		inputs = tokenizer(formatted_teencode, padding=True, truncation=True, return_tensors="pt")
		labels = inputs.input_ids.clone()
	
		# Huấn luyện đơn giản
		optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
		model.train()

		# Luyện data
		for epoch in range(3):
			outputs = model(**inputs, labels=labels)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			logging.info(f"Epoch {epoch+1}: Loss = {loss.item()}")
	
		logging.info("Train data teen code hoàn tất!")
		model.save_pretrained(model_path)
  
	logging.info("Đã train data teen code cho model!")
	
	train_data()
	train_teencode()
training_model()

logging.info("Đã train model!")

# Test
if __name__ == "__main__":
    check_model()
    training_model()