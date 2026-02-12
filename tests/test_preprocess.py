from src.data.preprocess import split_images

def test_split_images_ratio():
    dummy_images = [f"img_{i}.jpg" for i in range(100)]

    train, val, test = split_images(dummy_images, seed=42)

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Ensure no overlap
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
