Rummikub solver using old "joker locking" rules. Served directly by a single AWS Lambda function with a function URL. Can take a long time to load for the first time from lambda cold start, as it uses big Python libraries in a container.

Visit website: https://jeae5te7l4yzzmh7c3tjszhdwy0njroy.lambda-url.eu-west-1.on.aws.

## Useful commands

* `npm run build`   compile typescript to js
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `cdk deploy`      deploy this stack to your default AWS account/region
* `cdk diff`        compare deployed stack with current state
* `cdk synth`       emits the synthesized CloudFormation template
* `npm run build && cdk synth && cdk deploy`   build and deloy everything