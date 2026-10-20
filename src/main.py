import argparse
import asyncio

from orchestrator.orchestrator import Orchestrator

# Secretary system main contact interface
# Can change this up to a proper interface in the future

# Demo-level identity: the role is self-declared here and enforced by the
# Credit Lookup agent (see tools/creditLookupTool/credit_lookup/policy.py).
# A real login would replace these flags without changing anything else.
ROLE_LABELS = {"sales": "業務", "credit": "授信"}


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Orchestrator REPL.")
    parser.add_argument("--user", default="demo", help="user id recorded in the credit lookup audit log")
    parser.add_argument(
        "--role", choices=sorted(ROLE_LABELS), default="sales",
        help="sales sees contract status only; credit also sees balances, payment detail and risk notes",
    )
    return parser.parse_args()


async def main():

    args = parse_args()
    caller = {"user_id": args.user, "role": args.role}

    # Orchestrator.create() discovers remote A2A agents over the network
    # (see tool_router.py/remote_agent.py), which is why construction is now
    # async instead of a plain Orchestrator() call.
    orchestrator = await Orchestrator.create(caller)

    print("Welcome to the Agent-agent system!")  # Change this into something else in the future, maybe.
    print(f"Signed in as {args.user} ({ROLE_LABELS[args.role]} / {args.role}).")

    while True:

        # input() itself is still a blocking call -- fine here, since
        # nothing else needs to run concurrently while waiting on it.
        input_text = input(">>> ")

        if input_text.lower() == "bye":

            # Loop until the user gives a valid y/n answer, rather than
            # silently exiting or saving on unrecognized input.
            while True:
                choice = input("Save this chat's memory? (y/n): ").strip().lower()

                if choice == "y":

                    saved_path = orchestrator.end_session(save=True)
                    print(f"Chat memory saved to {saved_path}.")
                    break

                elif choice == "n":

                    orchestrator.end_session(save=False)
                    print("Chat memory discarded.")
                    break

                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            # Closes the persistent checkpointer's sqlite connection --
            # AsyncSqliteSaver's own docs warn the graph can hang on exit if
            # this isn't done (see Orchestrator.aclose()).
            await orchestrator.aclose()
            break

        print(await orchestrator.run_orchestrator(input_text))

if __name__ == "__main__":
    asyncio.run(main())
