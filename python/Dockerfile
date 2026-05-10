FROM public.ecr.aws/lambda/python:3.14

# Lambda task root is where code must live
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy project metadata first for better Docker layer caching
COPY pyproject.toml .

# Copy source code
COPY src ./src

# Install package + dependencies into Lambda task root
RUN pip install . --target "${LAMBDA_TASK_ROOT}"

# Lambda handler
CMD ["rummi_cube.handler.lambda_handler"]