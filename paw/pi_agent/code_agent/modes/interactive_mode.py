from __future__ import annotations

import os

from ..agent_session import AgentSession
from ..session_manager import SessionManager


def _print_help() -> None:
    print("/help     show commands")


def _print_help_schema(session: AgentSession) -> None:
    schema = session.get_command_schema()
    for group in schema["groups"]:
        print(f'group {group["id"]}: label={group["label"]} order={group["order"]}')
    for command in schema["commands"]:
        print(
            f'{command["name"]}:'
            f' group={command["group"]}'
            f' order={command["order"]}'
            f' usage={command["usage"]}'
            f' description={command["description"]}'
        )


def _print_tree(nodes: list[dict[str, object]], prefix: str = "") -> None:
    for node in nodes:
        entry = node["entry"]
        children = node["children"]
        marker = "*" if node.get("isLeaf") else "-"
        print(f"{prefix}{marker} {entry['id']} {entry['label']}")
        _print_tree(children, prefix + "  ")


async def run_interactive_mode(session: AgentSession) -> int:
    print("pi python port interactive mode")
    print("Type /help for commands.")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            return 0
        if message == "/help":
            _print_help_schema(session)
            continue
        if message == "/help brief":
            _print_help()
            continue
        if message == "/help schema":
            _print_help_schema(session)
            continue
        if message == "/selectors":
            registry = session.get_selector_registry()
            for group in registry["groups"]:
                print(f'group {group["id"]}: label={group["label"]} order={group["order"]}')
            for item in registry["selectors"]:
                print(
                    f'{item["id"]}:'
                    f' group={item["group"]}'
                    f' kind={item["kind"]}'
                    f' getter={item["getter"]}'
                    f' currentKey={item["currentKey"]}'
                    f' preview={session.get_selector(item["id"])["preview"]}'
                )
            continue
        if message.startswith("/selector "):
            selector_id = message.split(maxsplit=1)[1].strip()
            payload = session.get_selector(selector_id)
            selector = payload["selector"]
            print(
                f'{selector["id"]}:'
                f' group={selector["group"]}'
                f' kind={selector["kind"]}'
                f' getter={selector["getter"]}'
                f' currentKey={selector["currentKey"]}'
                f' preview={payload["preview"]}'
            )
            print(payload["data"])
            continue
        if message.startswith("/selector-item "):
            parts = message.split(maxsplit=2)
            if len(parts) < 3:
                print("usage: /selector-item SELECTOR_ID ITEM_ID")
                continue
            payload = session.get_selector_item(parts[1].strip(), parts[2].strip())
            print(
                f'{payload["selector"]["id"]}:'
                f' itemId={payload["requestedItemId"]}'
                f' resolvedItemId={payload["resolvedItemId"]}'
            )
            print(payload["item"])
            continue
        if message == "/session":
            stats = session.get_session_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
            continue
        if message == "/resources":
            schema = session.get_resource_schema()
            for key, value in schema["counts"].items():
                print(f"{key}: {value}")
            continue
        if message == "/resources schema":
            schema = session.get_resource_schema()
            for item in schema["skills"]:
                print(f'skill {item["name"]}: source={item["source"]} file={item["filePath"]}')
            for item in schema["prompts"]:
                print(f'prompt {item["name"]}: file={item["filePath"]}')
            for item in schema["themes"]:
                print(f'theme {item["name"]}: file={item["filePath"]}')
            for item in schema["extensions"]:
                print(f'extension {item["name"]}')
            for item in schema["agentsFiles"]:
                print(f"agents-file: {item}")
            continue
        if message.startswith("/resources item "):
            parts = message.split(maxsplit=3)
            if len(parts) < 4:
                print("usage: /resources item KIND ID")
                continue
            payload = session.get_resource_item(parts[2].strip(), parts[3].strip())
            print(
                f'{payload["kind"]}:'
                f' itemId={payload["requestedItemId"]}'
                f' resolvedItemId={payload["resolvedItemId"]}'
            )
            print(payload["item"])
            continue
        if message == "/packages schema":
            schema = session.get_package_selector_schema()
            for item in schema["items"]:
                print(
                    f'{item["source"]}:'
                    f' scope={item["scope"]}'
                    f' path={item["path"]}'
                    f' description={item["description"]}'
                )
            continue
        if message == "/packages":
            packages = session.list_packages()
            if not packages:
                print("No packages installed.")
            else:
                for package in packages:
                    print(f'{package["scope"]}: {package["source"]}')
                    print(f'  {package["path"]}')
            continue
        if message.startswith("/packages item "):
            source = message.split(maxsplit=2)[2].strip()
            payload = session.get_package_selector_item(source)
            print(
                f'{payload["selector"]["id"]}:'
                f' itemId={payload["requestedItemId"]}'
                f' resolvedItemId={payload["resolvedItemId"]}'
            )
            print(payload["item"])
            continue
        if message.startswith("/packages "):
            parts = message.split()
            if len(parts) < 2:
                print("usage: /packages [install|remove|uninstall|update] SOURCE [local]")
                continue
            command = parts[1].strip()
            source = parts[2].strip() if len(parts) > 2 else ""
            local = len(parts) > 3 and parts[3].strip().lower() in {"local", "project", "--local"}
            if command == "install":
                if not source:
                    print("usage: /packages install SOURCE [local]")
                    continue
                result = session.install_package(source, local=local)
                print(f'installed: {result["source"]} -> {result["path"]}')
            elif command in {"remove", "uninstall"}:
                if not source:
                    print("usage: /packages remove SOURCE [local]")
                    continue
                result = session.remove_package(source, local=local)
                if result["removed"]:
                    print(f'removed: {result["source"]}')
                else:
                    print(f'not found: {result["source"]}')
            elif command == "update":
                result = session.update_packages(source or None)
                updated = result["updated"]
                print("\n".join(updated) if updated else "No packages updated.")
            else:
                print("usage: /packages [item|install|remove|uninstall|update] SOURCE [local]")
            continue
        if message == "/settings":
            settings = session.get_settings_snapshot()
            for key, value in settings.items():
                print(f"{key}: {value}")
            continue
        if message == "/settings schema":
            schema = session.get_settings_schema()
            for group in schema.get("groups", []):
                print(f'group {group["id"]}: label={group["label"]} order={group["order"]}')
            for field in schema["fields"]:
                current = field.get("current")
                print(
                    f'{field["id"]}:'
                    f' order={field.get("order")}'
                    f' label={field.get("label")}'
                    f' group={field.get("group")}'
                    f' type={field["type"]}'
                    f' current={current}'
                )
                if field["type"] == "object":
                    for child in field.get("fields", []):
                        readonly = " readonly=True" if child.get("readonly") else ""
                        print(
                            f'  {child["id"]}:'
                            f' label={child.get("label")}'
                            f' type={child["type"]}'
                            f' updatePath={child.get("updatePath")}'
                            f' command={child.get("command")}{readonly}'
                        )
            continue
        if message.startswith("/settings "):
            parts = message.split()
            if len(parts) < 3:
                print("usage: /settings [name|model|theme|thinking|steering|followup|quiet|block-images|show-images|skill-commands|transport|retry|compaction] ...")
                continue
            if parts[1] == "name" and len(parts) >= 3:
                settings = session.update_settings({"sessionName": " ".join(parts[2:]).strip()})
            elif parts[1] == "model" and len(parts) == 3:
                settings = session.update_settings({"model": parts[2]})
            elif parts[1] == "theme" and len(parts) == 3:
                settings = session.update_settings({"theme": parts[2]})
            elif parts[1] == "thinking" and len(parts) == 3:
                settings = session.update_settings({"thinkingLevel": parts[2]})
            elif parts[1] == "steering" and len(parts) == 3:
                settings = session.update_settings({"steeringMode": parts[2]})
            elif parts[1] in {"followup", "follow-up"} and len(parts) == 3:
                settings = session.update_settings({"followUpMode": parts[2]})
            elif parts[1] == "quiet" and len(parts) == 3:
                settings = session.update_settings({"quietStartup": parts[2].strip().lower() in {"1", "true", "yes", "on"}})
            elif parts[1] == "block-images" and len(parts) == 3:
                settings = session.update_settings({"blockImages": parts[2].strip().lower() in {"1", "true", "yes", "on"}})
            elif parts[1] == "show-images" and len(parts) == 3:
                settings = session.update_settings({"showImages": parts[2].strip().lower() in {"1", "true", "yes", "on"}})
            elif parts[1] == "skill-commands" and len(parts) == 3:
                settings = session.update_settings({"enableSkillCommands": parts[2].strip().lower() in {"1", "true", "yes", "on"}})
            elif parts[1] == "transport" and len(parts) == 3:
                settings = session.update_settings({"transport": parts[2]})
            elif parts[1] == "retry" and len(parts) == 4:
                if parts[2] == "enabled":
                    settings = session.update_settings({"retry": {"enabled": parts[3].strip().lower() in {"1", "true", "yes", "on"}}})
                elif parts[2] == "max-retries":
                    settings = session.update_settings({"retry": {"maxRetries": int(parts[3])}})
                elif parts[2] == "base-delay-ms":
                    settings = session.update_settings({"retry": {"baseDelayMs": int(parts[3])}})
                elif parts[2] == "max-delay-ms":
                    settings = session.update_settings({"retry": {"maxDelayMs": int(parts[3])}})
                else:
                    print("usage: /settings retry [enabled|max-retries|base-delay-ms|max-delay-ms] VALUE")
                    continue
            elif parts[1] == "compaction" and len(parts) == 4:
                if parts[2] == "enabled":
                    settings = session.update_settings(
                        {"compaction": {"enabled": parts[3].strip().lower() in {"1", "true", "yes", "on"}}}
                    )
                elif parts[2] == "reserve":
                    settings = session.update_settings({"compaction": {"reserveTokens": int(parts[3])}})
                elif parts[2] == "keep":
                    settings = session.update_settings({"compaction": {"keepRecentTokens": int(parts[3])}})
                else:
                    print("usage: /settings compaction [enabled|reserve|keep] VALUE")
                    continue
            else:
                print("usage: /settings [name|model|theme|thinking|steering|followup|quiet|block-images|show-images|skill-commands|transport|retry|compaction] ...")
                continue
            print(f"settings: {settings}")
            continue
        if message == "/compaction":
            state = session.get_compaction_state()
            for key, value in state.items():
                print(f"{key}: {value}")
            continue
        if message == "/compaction schema":
            schema = session.get_compaction_schema()
            for field in schema["fields"]:
                readonly = " readonly=True" if field.get("readonly") else ""
                print(
                    f'{field["id"]}:'
                    f' order={field["order"]}'
                    f' label={field["label"]}'
                    f' type={field["type"]}'
                    f' current={field["current"]}'
                    f' updatePath={field.get("updatePath")}'
                    f' command={field.get("command")}{readonly}'
                )
            continue
        if message.startswith("/compaction "):
            parts = message.split()
            if len(parts) == 2 and parts[1] in {"enabled", "reserve", "keep"}:
                print(f"usage: /compaction {parts[1]} VALUE")
                continue
            if len(parts) != 3:
                print("usage: /compaction [enabled|reserve|keep] VALUE")
                continue
            if parts[1] == "enabled":
                state = session.set_compaction_enabled(parts[2].strip().lower() in {"1", "true", "yes", "on"})
            elif parts[1] == "reserve":
                state = session.set_compaction_reserve_tokens(int(parts[2]))
            elif parts[1] == "keep":
                state = session.set_compaction_keep_recent_tokens(int(parts[2]))
            else:
                print("usage: /compaction [enabled|reserve|keep] VALUE")
                continue
            print(
                "compaction:"
                f" enabled={state['enabled']}"
                f" estimated={state['estimatedTokens']}"
                f" threshold={state['thresholdTokens']}"
                f" reserve={state['reserveTokens']}"
                f" keep={state['keepRecentTokens']}"
                f" shouldCompact={state['shouldCompact']}"
            )
            continue
        if message == "/theme":
            themes = session.get_themes()
            print(f"theme: {themes['currentTheme']}")
            for item in themes["themes"]:
                marker = "*" if item["name"] == themes["currentTheme"] else "-"
                print(f"{marker} {item['name']} {item['filePath']}")
            continue
        if message == "/theme schema":
            schema = session.get_theme_selector_schema()
            print(f'currentTheme: {schema["currentTheme"]}')
            for item in schema["items"]:
                print(
                    f'{item["id"]}:'
                    f' position={item["position"]}'
                    f' isCurrent={item["isCurrent"]}'
                    f' label={item["label"]}'
                    f' description={item["description"]}'
                )
            continue
        if message.startswith("/theme "):
            theme = message.split(maxsplit=1)[1].strip()
            themes = session.set_theme(theme)
            print(f"theme: {themes['currentTheme']}")
            continue
        if message.startswith("/model"):
            parts = message.split(maxsplit=1)
            if len(parts) == 1:
                print(f"{session.model.provider}/{session.model.id}")
            elif parts[1].strip() == "schema":
                schema = session.get_model_selector_schema()
                print(f'currentModel: {schema["currentModel"]["provider"]}/{schema["currentModel"]["id"]}')
                for item in schema["items"]:
                    print(
                        f'{item["id"]}:'
                        f' position={item["position"]}'
                        f' isCurrent={item["isCurrent"]}'
                        f' label={item["label"]}'
                        f' description={item["description"]}'
                    )
            else:
                await session.set_model("openai", parts[1].strip())
                print(f"model: {session.model.provider}/{session.model.id}")
            continue
        if message == "/cycle-model":
            await session.cycle_model()
            print(f"model: {session.model.provider}/{session.model.id}")
            continue
        if message.startswith("/thinking"):
            parts = message.split(maxsplit=1)
            if len(parts) == 1:
                print(f"thinking: {session.thinking_level}")
            elif parts[1].strip() == "schema":
                schema = session.get_thinking_selector_schema()
                print(f'currentThinkingLevel: {schema["currentThinkingLevel"]}')
                for item in schema["items"]:
                    print(
                        f'{item["id"]}:'
                        f' position={item["position"]}'
                        f' isCurrent={item["isCurrent"]}'
                        f' isSupported={item["isSupported"]}'
                        f' label={item["label"]}'
                    )
            elif parts[1].strip() == "cycle":
                print(f"thinking: {session.cycle_thinking_level()}")
            else:
                session.set_thinking_level(parts[1].strip())
                print(f"thinking: {session.thinking_level}")
            continue
        if message.startswith("/steering"):
            parts = message.split(maxsplit=1)
            if len(parts) == 1:
                print(f"steering: {session.get_state().steering_mode}")
            elif parts[1].strip() == "schema":
                schema = session.get_steering_selector_schema()
                print(f'currentSteeringMode: {schema["currentSteeringMode"]}')
                for item in schema["items"]:
                    print(
                        f'{item["id"]}:'
                        f' position={item["position"]}'
                        f' isCurrent={item["isCurrent"]}'
                        f' label={item["label"]}'
                        f' description={item["description"]}'
                    )
            else:
                print(f"steering: {session.set_steering_mode(parts[1].strip())}")
            continue
        if message.startswith("/followup") or message.startswith("/follow-up"):
            parts = message.split(maxsplit=1)
            if len(parts) == 1:
                print(f"follow-up: {session.get_state().follow_up_mode}")
            elif parts[1].strip() == "schema":
                schema = session.get_follow_up_selector_schema()
                print(f'currentFollowUpMode: {schema["currentFollowUpMode"]}')
                for item in schema["items"]:
                    print(
                        f'{item["id"]}:'
                        f' position={item["position"]}'
                        f' isCurrent={item["isCurrent"]}'
                        f' label={item["label"]}'
                        f' description={item["description"]}'
                    )
            else:
                print(f"follow-up: {session.set_follow_up_mode(parts[1].strip())}")
            continue
        if message.startswith("/name"):
            parts = message.split(maxsplit=1)
            session.set_session_name(parts[1].strip() if len(parts) > 1 else None)
            print(f"session name: {session.session_manager.get_session_name()}")
            continue
        if message == "/new":
            await session.new_session()
            print("new session")
            continue
        if message == "/reload":
            await session.resource_loader.reload()
            print("resources reloaded")
            continue
        if message == "/resume" or message.startswith("/resume "):
            query = message.split(maxsplit=1)[1].strip() if " " in message else ""
            if not query:
                session_file = SessionManager.find_most_recent_session(os.getcwd())
                if session_file is None:
                    print("no previous session")
                else:
                    await session.switch_session(str(session_file))
                    print(f"resumed: {session_file}")
            else:
                resolved = await session.resolve_and_switch_session(query)
                print(f"resumed: {resolved}")
            continue
        if message == "/sessions":
            session_infos = session.list_session_infos()
            if not session_infos:
                print("no previous sessions")
            else:
                for info in session_infos:
                    name = f" [{info.name}]" if info.name else ""
                    print(f"{info.path}{name}")
                    print(f"  modified={info.modified} messages={info.message_count} first={info.first_message}")
            continue
        if message == "/sessions schema":
            schema = session.get_session_selector_schema()
            print(f'currentSessionFile: {schema["currentSessionFile"]}')
            for item in schema["items"]:
                print(
                    f'{item["path"]}:'
                    f' id={item["id"]}'
                    f' position={item["position"]}'
                    f' isCurrent={item["isCurrent"]}'
                    f' messageCount={item["messageCount"]}'
                    f' label={item["label"]}'
                    f' description={item["description"]}'
                )
            continue
        if message.startswith("/switch "):
            query = message.split(maxsplit=1)[1].strip()
            resolved = await session.resolve_and_switch_session(query)
            print(f"switched: {resolved}")
            continue
        if message == "/last":
            print(session.get_last_assistant_text() or "")
            continue
        if message == "/tree":
            tree = session.get_tree()
            if not tree:
                print("empty session tree")
            else:
                _print_tree(tree)
            continue
        if message == "/tree schema":
            schema = session.get_tree_schema()
            print(f'currentLeafId: {schema["currentLeafId"]}')
            for item in schema["items"]:
                print(
                    f'{item["id"]}:'
                    f' parentId={item["parentId"]}'
                    f' depth={item["depth"]}'
                    f' childCount={item["childCount"]}'
                    f' isLeaf={item["isLeaf"]}'
                    f' isOnCurrentBranch={item["isOnCurrentBranch"]}'
                    f' label={item["label"]}'
                )
            continue
        if message.startswith("/tree "):
            entry_id = message.split(maxsplit=1)[1].strip()
            result = session.branch(entry_id if entry_id else None)
            print(f"branched: {result['leafId'] or 'root'}")
            continue
        if message.startswith("/branch-summary "):
            parts = message.split(maxsplit=2)
            if len(parts) < 3:
                print("usage: /branch-summary ID SUMMARY")
            else:
                result = session.branch_with_summary(parts[1].strip(), parts[2].strip())
                print(f"branch summary: {result['summaryEntryId']}")
            continue
        if message.startswith("/branch-summary-auto "):
            parts = message.split(maxsplit=2)
            entry_id = parts[1].strip() if len(parts) > 1 else ""
            instructions = parts[2].strip() if len(parts) > 2 else None
            result = await session.auto_branch_with_summary(entry_id, instructions)
            print(f"branch summary: {result['summaryEntryId']}")
            continue
        if message.startswith("/compact "):
            parts = message.split(maxsplit=3)
            if len(parts) < 4:
                print("usage: /compact FIRST_KEPT_ID TOKENS SUMMARY")
            else:
                result = session.compact(parts[1].strip(), parts[3].strip(), int(parts[2]))
                print(f"compaction: {result['compactionEntryId']}")
            continue
        if message.startswith("/compact-auto "):
            parts = message.split(maxsplit=2)
            if len(parts) < 2:
                print("usage: /compact-auto FIRST_KEPT_ID [INSTRUCTIONS]")
            else:
                result = await session.auto_compact(parts[1].strip(), parts[2].strip() if len(parts) > 2 else None)
                print(f"compaction: {result['compactionEntryId']}")
            continue
        if message == "/summarize" or message.startswith("/summarize "):
            instructions = message.split(maxsplit=1)[1].strip() if " " in message else None
            result = await session.generate_summary(instructions)
            print(result["summary"])
            continue
        if message == "/fork":
            messages = session.get_fork_messages()
            if not messages:
                print("no user messages available for fork")
            else:
                for item in messages:
                    print(f'{item["entryId"]}: {item["text"]}')
            continue
        if message.startswith("/fork "):
            entry_id = message.split(maxsplit=1)[1].strip()
            result = await session.fork(entry_id)
            print(f'forked: {result["selectedText"]}')
            continue
        chunks: list[str] = []

        def listener(event) -> None:
            if getattr(event, "type", None) == "message_update":
                assistant_event = getattr(event, "assistant_message_event", None)
                if getattr(assistant_event, "type", None) == "text_delta":
                    chunks.append(assistant_event.delta)

        unsubscribe = session.subscribe(listener)
        try:
            await session.prompt(message)
        finally:
            unsubscribe()
        print("".join(chunks).strip())
