from unittest import TestCase

from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent

from rummi_cube.handler import lambda_handler


def get_test_event(path: str, table: list[str], rack: str, strategy: str, method="GET"):
    return LambdaFunctionUrlEvent(data={
        "rawPath": path,
        "queryStringParameters": {
            "table": "\n".join(table),
            "rack": rack,
            "strategy": strategy,
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
        response = lambda_handler(get_test_event("", [], "", "", method="POST"), None)

        self.assertEqual(response["statusCode"], 400)
        self.assertIn("Invalid method", response["body"])

    def test_entry(self):
        response = lambda_handler(get_test_event("/solve", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10", "entry"), None)
        print(response)
        self.assertEqual(response["statusCode"], 200, response)

    def test_maximize_value(self):
        response = lambda_handler(
            get_test_event("/solve", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10", "maximize_value"), None)

        self.assertEqual(response["statusCode"], 200, response)

    def test_place_minimum(self):
        response = lambda_handler(get_test_event("/solve", ["a13 b13 y13", "a1 a2 a3"], "a10 b10 r10", "place_minimum"),
                                  None)

        self.assertEqual(response["statusCode"], 200, response)
