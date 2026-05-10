import { Construct } from "constructs";
import { DockerImageCode, DockerImageFunction, FunctionUrlAuthType } from "aws-cdk-lib/aws-lambda";
import { Duration, Stack, StackProps } from "aws-cdk-lib";
import path from "path";

export class RummiLambdaStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const lambda = new DockerImageFunction(this, 'RummiCubeLambda', {
      functionName: "RummiCube",
      code: DockerImageCode.fromImageAsset(
        path.join(__dirname, "..", "python")
      ),
      memorySize: 1000,
      reservedConcurrentExecutions: 1,
      timeout: Duration.seconds(10),
    });

    lambda.addFunctionUrl({authType: FunctionUrlAuthType.NONE});
  }
}