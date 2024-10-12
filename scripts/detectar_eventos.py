from bibmon.models.generic_model import GenericModel
from bibmon.utils.preprocess import PreProcess
import Detection 


preprocessor = PreProcess()
preprocessed_data = preprocessor.remove_nan_observations(df)

model = GenericModel(model_type='random_forest')  


model.train(preprocessed_data)


detections = model.predict(preprocessed_data)


print(detections.head())

