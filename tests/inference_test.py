import time

import pytest
import torch

from chatspace import ChatSpace

cuda_param = "cuda" if torch.cuda.is_available() else pytest.param("cuda", marks=pytest.mark.skip)


@pytest.fixture(params=["cpu", cuda_param])
def spacer(request):
    device = torch.device(request.param)
    return ChatSpace(device=device)


def check_speed(spacer, text):
    start_time = time.time()
    output = spacer.space(text)
    end_time = time.time()
    print(f"origin: {text}\tfixed: {output}\ttime: {end_time - start_time}sec")
    return output, start_time - end_time


def test_space(spacer):
    assert spacer.space("안녕내이름은뽀로로야") == "안녕 내 이름은 뽀로로야"
    assert spacer.space("만나서반가워!") == "만나서 반가워!"


def test_custom_vocab(spacer):
    assert spacer.space("행신역 앞 피자마루") == "행신역 앞 피자 마루"
    assert spacer.space("행신역 앞 피자마루", custom_vocab=["피자마루"]) == "행신역 앞 피자마루"


def test_time(spacer):
    output, speed = check_speed(spacer, "안녕 내이름은 뽀로로야")
    assert speed < 0.1

    output, speed = check_speed(spacer, "만나서 반가워!")
    assert speed < 0.1
