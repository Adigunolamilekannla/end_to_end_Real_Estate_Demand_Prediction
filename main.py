from src.components.data_injection import DataInjection, DataInjectionConfig

# 1. Create configuration
data_ingestion_config = DataInjectionConfig()

# 2. Initialize the component with config
data_ingestion = DataInjection()

# 3. Start data ingestion and collect the artifact
data_ingestion_artifact = data_ingestion.initiate_data_injection()
