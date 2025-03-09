# Import
import os, logging, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import test
if __name__ == "__main__":
	from Debug.Logging import loggingConfig
	loggingConfig()

# Cấu hình model
model_path = "d:/AI_Model/GPT2_Model"
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check model
def check_model():
	logging.info("Kiem tra model...")

	if not os.path.exists(model_path):
		logging.warning("Không có model, đang tải...")

		os.makedirs(model_path, exist_ok=True)
		try:
			model = AutoModelForCausalLM.from_pretrained(model_name)
			tokenizer = AutoTokenizer.from_pretrained(model_name)
			model.save_pretrained(model_path)
			tokenizer.save_pretrained(model_path)
			logging.info(f"Model đã được tải vào: {model_path}")
   
		except Exception as e:
			logging.error(f"Lỗi tải model: {e}")
			return

	else:
		logging.info(f"Model đã có trong: {model_path}")

	logging.info("Đã kiểm tra model!")


# Training model
def training_model():
	# Load model và tokenizer
	model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	if torch.cuda.is_available():
		model.half() # Giảm model

	# Train data
	def train_data():
		logging.info("Bắt đầu train data...")

		# Load Data hội thoại
		data_path = "Memory/Data_training.json"
		with open(data_path, "r", encoding="utf-8") as f:
			dataset = json.load(f)

		# Xử lý form data
		formatted_data = []
		for item in dataset:
			if "input" in item and "output" in item:
				all_inputs = " | ".join(item["input"]) if isinstance(item["input"], list) else item["input"]
				if isinstance(item["output"], list):
					for out in item["output"]:
						formatted_data.append(f"{all_inputs} => {out}")
		
				else:
					formatted_data.append(f"{all_inputs} => {item['output']}")

		# Data tokenizer
		inputs = tokenizer(formatted_data, padding=True, truncation=True, return_tensors="pt")
		if device == "cuda":
			inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}
		labels = inputs["input_ids"].clone()

		# Huấn luyện đơn giản
		optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
		model.train()

		# Luyện data
		for epoch in range(3):
			outputs = model(**inputs, labels=labels)
			loss = outputs.loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			logging.info(f"Epoch {epoch+1}: Loss = {loss.item()}")

		logging.info("Train data hội thoại hoàn tất!")

	# Train teencode
	def train_teencode():
		logging.info("Bắt đầu train data teen code...")

		# Load Data teen code
		slang_data_path = "Memory/Data_teencode.json"
		with open(slang_data_path, "r", encoding="utf-8") as f:
			dataset = json.load(f)

		# Format dữ liệu
		formatted_teencode = [
			f"{key} => {value}" for key, value in dataset.items()
			]

		# Data tokenizer
		inputs = tokenizer(formatted_teencode, padding=True, truncation=True, return_tensors="pt")
		if device == "cuda":
			inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}
		labels = inputs["input_ids"].clone()

		# Huấn luyện đơn giản
		optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
		model.train()

		# Luyện data
		for epoch in range(3):
			outputs = model(**inputs, labels=labels)
			loss = outputs.loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			logging.info(f"Epoch {epoch+1}: Loss = {loss.item()}")

		logging.info("Train data teen code hoàn tất!")

	with torch.no_grad(): # Chạy từng phân
		train_data()
		train_teencode()
  
	# Lưu model
	model.save_pretrained(model_path)
training_model()

logging.info("Đã train model!")

# Test
if __name__ == "__main__":
	check_model()
