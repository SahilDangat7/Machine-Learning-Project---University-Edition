from wilds import get_dataset

# Download labelled data only
dataset = get_dataset(dataset="poverty", download=True)

train_data = dataset.get_subset("train")
val_data   = dataset.get_subset("val")
test_data  = dataset.get_subset("test")

print(f"Train size : {len(train_data)}")
print(f"Val size   : {len(val_data)}")
print(f"Test size  : {len(test_data)}")

x, y, metadata = train_data[0]
print(f"Image shape : {x.shape}")
print(f"Label       : {y.item():.4f}")
print(f"Metadata    : {metadata}")