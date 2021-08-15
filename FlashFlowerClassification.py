import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# 1. Use the datastore 

datamodule = ImageClassificationData.from_folders(
    train_folder="/home/jovyan/flower-dataset/train/data/hymenoptera_data/train/",
    val_folder="/home/jovyan/flower-dataset/val/",
    test_folder="/home/jovyan/flower-dataset/test/",
)

# 2. Build the task
model = ImageClassifier(num_classes=datamodule.num_classes)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")
 
trainer.test()


# 4. Predict what's on a few images! ants or bees?
model.serializer = Labels()

predictions = model.predict(
    [
        "/home/jovyan/flower-dataset/val/roses/4644336779_acd973528c.jpg",
        "/home/jovyan/flower-dataset/val/tulips/2436998042_4906ea07af.jpg",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
