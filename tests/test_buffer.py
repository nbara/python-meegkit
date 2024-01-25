import numpy as np
import pytest

from meegkit.utils.buffer import Buffer


def test_push():
    buf = Buffer(32, 1)
    data = np.arange(1000)
    buf.push(data[:10])
    assert buf.counter == 10
    assert buf.head == 10
    assert buf.tail == 0

def test_get_new_samples():
    buf = Buffer(32, 1)
    data = np.arange(1000)
    buf.push(data[:10])
    samples = buf.get_new_samples(5)
    np.testing.assert_array_equal(samples.flatten(), data[:5])
    assert buf.tail == 5
    samples = buf.get_new_samples(5)
    np.testing.assert_array_equal(samples.flatten(), data[5:10])

def test_reset():
    buf = Buffer(32, 1)
    data = np.arange(1000)
    buf.push(data[:10])
    buf.reset()
    assert buf.counter == 0
    assert buf.head == 0
    assert buf.tail == 0

def test_repr():
    buf = Buffer(32, 1)
    repr = buf.__repr__()
    assert repr == "Buffer(32, 1)\n> counter: 0\n> head: 0\n> tail: 0\n"

def test_overflow_warning():
    buf = Buffer(32, 1)
    data = np.arange(1000)
    with pytest.warns(UserWarning, match="Buffer overflow: some old data has been lost"):
        buf.push(data)

def test_repeated_push():
    buf = Buffer(32, 1)
    data = np.arange(1000)

    print(buf)
    buf.push(data[:10])
    print(buf)
    buf.push(data[10:20])
    print(buf)

    with pytest.warns(UserWarning, match="Buffer overflow: some old data has been discarded"):
        buf.push(data[20:40])
        print(buf)
        assert buf.counter == 40
        assert buf.head == 8
        assert buf._head == 40
        assert buf.tail == 8
        assert buf._tail == 8

    out = buf.get_new_samples()
    print(out.flatten())
    np.testing.assert_array_equal(out.flatten(), data[8:40]) # first 8 samples are lost


if __name__ == "__main__":
    # test_get_new_samples()
    # test_repeated_push()
    pytest.main([__file__])