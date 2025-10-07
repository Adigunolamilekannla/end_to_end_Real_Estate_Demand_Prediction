from src.components.data_injection import DataInjection, DataInjectionConfig
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

# 1. Create configuration
#data_ingestion_config = DataInjectionConfig()

# 2. Initialize the component with config
data_ingestion = DataInjection()

# 3. Start data ingestion and collect the artifact
data_ingestion_artifact = data_ingestion.initiate_data_injection()

#data_trans_config = DataTransformationConfig()

data_transformation = DataTransformation()
data_transformation.initiate_transform_data()

#model_trainer_config = ModelTrainerConfig()

model_trainer = ModelTrainer()
model_trainer.initiate_train_model()



