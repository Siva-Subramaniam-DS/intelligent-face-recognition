import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import numpy as np

class SiameseNetwork:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self.build_model()
        
    def build_base_network(self):
        """Build the base CNN network for feature extraction"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False
        
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        
        return Model(inputs=base_model.input, outputs=x)
    
    def euclidean_distance(self, vectors):
        """Compute the Euclidean distance between two vectors"""
        x, y = vectors
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))
    
    def build_model(self):
        """Build the complete Siamese network"""
        base_network = self.build_base_network()
        
        # Input layers
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        # Get feature vectors
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Compute distance
        distance = Lambda(self.euclidean_distance)([processed_a, processed_b])
        
        # Output layer
        output = Dense(1, activation='sigmoid')(distance)
        
        # Create model
        model = Model([input_a, input_b], output)
        
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model with appropriate loss and metrics"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_data, validation_data, epochs=10, batch_size=32):
        """Train the model"""
        X1_train, X2_train, y_train = train_data
        X1_val, X2_val, y_val = validation_data
        
        history = self.model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            batch_size=batch_size,
            epochs=epochs
        )
        
        return history
    
    def predict(self, image_pair):
        """Predict similarity for a pair of images"""
        return self.model.predict(image_pair)
    
    def get_embedding(self, image):
        """Get the embedding for a single image"""
        base_network = self.build_base_network()
        return base_network.predict(np.expand_dims(image, axis=0))
    
    def save_model(self, path):
        """Save the model to disk"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a model from disk"""
        self.model = tf.keras.models.load_model(path) 