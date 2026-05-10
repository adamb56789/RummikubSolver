from unittest import TestCase

from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent

from rummi_cube.handler import lambda_handler


def get_test_event(path: str, table: list[str], rack: str, method="GET"):
    return LambdaFunctionUrlEvent(data={
        "rawPath": path,
        "queryStringParameters": {
            "table": ",".join(table),
            "rack": rack
        },
        "requestContext": {
            "http": {
                "method": method,
                "sourceIp": "123.123.123.123",
                "userAgent": "agent"
            },
            "requestId": "id",
        },
        "body": "Hello!",
    })


class TestLambdaHandler(TestCase):

    def test_rejects_wrong_method(self):
        response = lambda_handler(get_test_event("", [], "", method="POST"), None)

        self.assertEqual(response["statusCode"], 400)
        self.assertEqual(response["body"], "Invalid method")

    def test_entry(self):
        response = lambda_handler(get_test_event("/entry", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10"), None)

        self.assertEqual(response["statusCode"], 200, response)

    def test_maximize_value(self):
        response = lambda_handler(get_test_event("/maximize-value", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10"), None)

        self.assertEqual(response["statusCode"], 200, response)

    def test_place_minimum(self):
        response = lambda_handler(get_test_event("/place-minimum", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10"), None)

        self.assertEqual(response["statusCode"], 200, response)

    def test_idk(self):
        event = get_test_event("/maximize-value", [], "y11 r5 a6 b13 y2 a12 y7 r4 y13 a13")
        event["queryStringParameters"]["table"] = "r10 J0 r12 r13,r4 r5 J0,a12 r12 y12,a7 b7 y7,b4 b5 b6,a12 b12 r12,r7 r8 r9 r10 r11,a9 a10 a11,b2 b3 b4 b5 b6,b7 b8 b9 b10,y5 y6 y7 y8,b3 r3 y3"
        response = lambda_handler(event, None)

        self.assertEqual(response["statusCode"], 200, response)