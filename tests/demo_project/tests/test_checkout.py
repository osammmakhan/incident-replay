"""Tests for the CheckoutService (partial coverage — no None discount test)."""
from app.checkout import CheckoutService


class FakeOrder:
    def __init__(self, discount, subtotal=100.0):
        self.discount = discount
        self.subtotal = subtotal


def test_calculate_discount_zero():
    svc = CheckoutService()
    order = FakeOrder(discount=0.0)
    assert svc.calculate_discount(order) == 0.0


def test_calculate_discount_half():
    svc = CheckoutService()
    order = FakeOrder(discount=0.3)
    assert svc.calculate_discount(order) == 0.3


def test_calculate_discount_capped():
    svc = CheckoutService()
    order = FakeOrder(discount=0.9)
    assert svc.calculate_discount(order) == 0.5


def test_total():
    svc = CheckoutService()
    order = FakeOrder(discount=0.2, subtotal=100.0)
    assert svc.total(order) == 80.0
    # NOTE: no test for discount=None — that gap is the incident scenario.
