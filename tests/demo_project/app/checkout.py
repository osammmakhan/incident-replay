"""
Demo checkout module with a deliberate null-safety bug.

The bug: ``calculate_discount`` calls ``min(order.discount, ...)`` without
checking whether ``order.discount`` is ``None``.  When a customer omits the
discount field, the request fails with::

    TypeError: '<' not supported between instances of 'NoneType' and 'float'

This file is intentionally kept minimal so that the incident-replay
orchestration tests can exercise real code-agent heuristics without any
third-party dependency.
"""


class CheckoutService:
    MAX_DISCOUNT_RATE = 0.5

    def calculate_discount(self, order):
        """Return the capped discount rate for *order*.

        BUG: order.discount may be None when the field is omitted from the
        request payload; the min() call raises TypeError in that case.
        """
        discount_rate = max(0.0, min(order.discount, self.MAX_DISCOUNT_RATE))
        return discount_rate

    def total(self, order):
        """Return the order total after applying the discount."""
        rate = self.calculate_discount(order)
        return order.subtotal * (1.0 - rate)
