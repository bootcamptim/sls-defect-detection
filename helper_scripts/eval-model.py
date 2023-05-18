import os
import tensorflow as tf
from tensorflow.keras import metrics

model_dir = '../models/dense3_softmax/particledrag_dataset=230517_epochs=10_WeightedCategoricalCrossentropy.h5'
# model_dir = '../models/dense3_softmax/particledrag_dataset=230503_epochs=15.h5'
data_dir = '../230517_Consolidated Data'

eval_dir = '../eval-data'
eval_data = tf.keras.utils.image_dataset_from_directory(eval_dir, image_size=(480, 640), label_mode='categorical')

# Compute class weights and custom loss function
total_count = len(os.listdir(data_dir))
ok_count = len(os.listdir(os.path.join(data_dir, 'ok')))
defect_count = len(os.listdir(os.path.join(data_dir, 'defect')))

weights = tf.constant([total_count/ok_count, total_count/defect_count])

# Define custom loss function
def weighted_cross_entropy(y_true, y_pred):
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    y_true_class = tf.argmax(y_true, axis=1)
    weights_ce = tf.gather(weights, y_true_class)
    weighted_ce = ce * weights_ce
    
    return tf.reduce_mean(weighted_ce)

# Custom F1 metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Load model
model = tf.keras.models.load_model(model_dir, compile=False)

# Compile the model with custom loss function and metrics
model.compile(
    optimizer='adam', 
    loss=weighted_cross_entropy, 
    metrics=[
        'accuracy', 
        metrics.Precision(name='precision'),
        F1Score()
    ]
)


# Evaluate the model
results = model.evaluate(eval_data)
print('Loss:', results[0])
print('Accuracy:', results[1])
print('Precision:', results[2])
print('F1 Score:', results[3])
