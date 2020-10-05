    """
model.add(tf.keras.layers.Conv2D(256, (3, 3), input_shape=imgs.shape[1:]))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(256, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
"""



"""
model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=imgs.shape[1:]))
        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(256, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
                
        model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))    
"""





"""
        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=imgs.shape[1:]))
        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(256, (3, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5))
                
        model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
"""


