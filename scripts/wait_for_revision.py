"""Wait for the deployed Azure revision itself before running the public HTTP smoke test."""

import argparse
import json
import math
import subprocess
import sys
import time


def read_revision(app_name, resource_group, revision):
    result = subprocess.run(
        ["az", "containerapp", "revision", "show", "--name", app_name,
         "--resource-group", resource_group, "--revision", revision,
         "--output", "json", "--only-show-errors"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(result.stdout)


def wait_for_revision(app_name, resource_group, revision, image, timeout):
    deadline = time.monotonic() + timeout
    while True:
        properties = read_revision(app_name, resource_group, revision)["properties"]
        containers = properties.get("template", {}).get("containers", [])
        health = properties.get("healthState")
        provisioning = properties.get("provisioningState")
        running = properties.get("runningState")
        print(f"Revision {revision}: provisioning={provisioning}, health={health}, "
              f"running={running}", flush=True)
        if provisioning == "Failed" or running == "Failed":
            raise ValueError("The deployed revision failed; check its Azure logs")
        if (properties.get("active") is True and provisioning == "Provisioned"
                and health == "Healthy" and running == "Running"
                and any(container.get("image") == image for container in containers)):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("The deployed image did not become healthy before the deadline")
        time.sleep(min(15, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not args.revision.strip():
        parser.error("Azure did not return a deployed revision name")
    try:
        wait_for_revision(args.app_name, args.resource_group, args.revision,
                          args.image, args.timeout)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        # CLI output can contain environment variables; never echo its stdout or stderr.
        detail = str(exc) if type(exc) is ValueError else type(exc).__name__
        print(f"Revision check failed: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
