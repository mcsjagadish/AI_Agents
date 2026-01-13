import asyncio
from typing_extensions import Never
from agent_framework import WorkflowBuilder, WorkflowContext, WorkflowOutputEvent, executor

class UpperCase(Executor):

    def __init__(self, id:str):
        super().__init__(id=id)

    
    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """Convert the input to uppercase and forward it to the next node.

        Note: The WorkflowContext is parameterized with the type this handler will
        emit. Here WorkflowContext[str] means downstream nodes should expect str.
        """
        result = text.upper()

        # Send the result to the next executor in the workflow.
        await ctx.send_message(result)

    @executor(id="reverse_text_executor")
    async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
        """Reverse the input and yield the workflow output."""
        result = text[::-1]

        # Yield the final output for this workflow run
        await ctx.yield_output(result)


upper_case = UpperCase(id="upper_case_executor")

workflow = (
    WorkflowBuilder()
    .add_edge(upper_case, upper_case.reverse_text)
    .set_start_executor(upper_case)
    .build()
)

async def main():
    # Run the workflow and stream events
    async for event in workflow.run_stream("hello world"):
        print(f"Event: {event}")
        if isinstance(event, WorkflowOutputEvent):
            print(f"Workflow completed with result: {event.data}")

if __name__ == "__main__":
    asyncio.run(main())