import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 256, size=(70000, 784)).astype(float)
    y = rng.integers(0, 10, size=70000).astype(str)
    return X, y


@pytest.fixture
def downsampled_data(raw_data):
    X, y = raw_data
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=5000, replace=False)
    return X[indices], y[indices], indices


@pytest.fixture
def split_data(downsampled_data):
    X, y, _ = downsampled_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --- Downsampling ---

def test_downsample_size(downsampled_data):
    _, _, indices = downsampled_data
    assert len(indices) == 5000


def test_downsample_no_duplicates(downsampled_data):
    _, _, indices = downsampled_data
    assert len(indices) == len(set(indices))


# --- Train/Test Split ---

def test_split_ratio(split_data):
    X_train, X_test, _, _ = split_data
    assert len(X_train) == 4000
    assert len(X_test) == 1000


def test_split_no_overlap(downsampled_data):
    X, y, _ = downsampled_data
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    assert set(train_idx).isdisjoint(set(test_idx))


# --- StandardScaler ---

def test_scaler_train_mean_and_std(split_data):
    X_train, _, _, _ = split_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    np.testing.assert_allclose(X_scaled.mean(), 0, atol=0.1)
    np.testing.assert_allclose(X_scaled.std(), 1, atol=0.1)


def test_scaler_test_uses_train_stats(split_data):
    X_train, X_test, _, _ = split_data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    # test set scaled with train mean/std won't have mean exactly 0
    expected = (X_test - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(X_test_scaled, expected)


def test_scaler_raises_before_fit():
    scaler = StandardScaler()
    X = np.random.rand(100, 10)
    with pytest.raises(Exception):
        scaler.transform(X)
