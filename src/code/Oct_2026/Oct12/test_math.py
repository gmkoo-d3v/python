def add(a, b):
    return a + b

# pytest 는 test_로 시작하는 함수를 자동으로 찾아 실행한다

def test_add_positive():
    assert add(2, 3) == 5

def test_add_zero():
    assert add(0, 0) == 0

def test_add_negative():
    assert add(-1, -1) == -2
    