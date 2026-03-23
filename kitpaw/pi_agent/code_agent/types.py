from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..agent.types import AgentMessage, ThinkingLevel
from ..ai.types import ImageContent, Model

QueueMode = Literal["all", "one-at-a-time"]
Transport = Literal["sse", "websocket", "auto"]


@dataclass(slots=True)
class PackageSource:
    source: str
    scope: Literal["user", "project"] = "user"


@dataclass(slots=True)
class RetrySettings:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000
    max_delay_ms: int = 60_000


@dataclass(slots=True)
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass(slots=True)
class ImageSettings:
    auto_resize: bool = True
    block_images: bool = False


@dataclass(slots=True)
class TerminalSettings:
    show_images: bool = True
    clear_on_shrink: bool = False


@dataclass(slots=True)
class Settings:
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking_level: ThinkingLevel | None = None
    theme: str = "dark"
    quiet_startup: bool = False
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    transport: Transport = "sse"
    shell_command_prefix: str | None = None
    shell_path: str | None = None
    enabled_models: list[str] = field(default_factory=list)
    packages: list[PackageSource | str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    enable_skill_commands: bool = True
    retry: RetrySettings = field(default_factory=RetrySettings)
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    images: ImageSettings = field(default_factory=ImageSettings)
    terminal: TerminalSettings = field(default_factory=TerminalSettings)


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: Literal["global", "project", "settings", "cli", "package"] = "project"
    disable_model_invocation: bool = False


@dataclass(slots=True)
class PromptTemplate:
    name: str
    text: str
    file_path: str


@dataclass(slots=True)
class ThemeResource:
    name: str
    file_path: str
    data: dict[str, Any]


@dataclass(slots=True)
class ResourceDiagnostic:
    type: Literal["warning", "error", "collision"]
    message: str


@dataclass(slots=True)
class LoadedSkills:
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


@dataclass(slots=True)
class LoadedPrompts:
    prompts: list[PromptTemplate] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


@dataclass(slots=True)
class LoadedThemes:
    themes: list[ThemeResource] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


@dataclass(slots=True)
class LoadedAgentsFiles:
    agents_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadedExtensions:
    extensions: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionStateSnapshot:
    model: Model | None
    thinking_level: ThinkingLevel
    is_streaming: bool
    session_file: str | None
    session_id: str
    message_count: int
    steering_mode: QueueMode
    follow_up_mode: QueueMode
    pending_message_count: int


@dataclass(slots=True)
class SessionInfo:
    path: str
    id: str
    cwd: str
    name: str | None
    parent_session_path: str | None
    created: str
    modified: str
    message_count: int
    first_message: str
    all_messages_text: str


@dataclass(slots=True)
class PromptOptions:
    expand_prompt_templates: bool = True
    images: list[ImageContent] | None = None
    streaming_behavior: Literal["steer", "followUp"] | None = None
    source: str = "interactive"


@dataclass(slots=True)
class AgentSessionEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionEntryBase:
    type: str
    id: str
    parent_id: str | None
    timestamp: str


@dataclass(slots=True)
class SessionHeader:
    type: Literal["session"] = "session"
    version: int = 3
    id: str = ""
    timestamp: str = ""
    cwd: str = ""
    parent_session: str | None = None


@dataclass(slots=True)
class SessionMessageEntry(SessionEntryBase):
    message: AgentMessage | dict[str, Any] = field(default_factory=dict)


CURRENT_SESSION_VERSION = 3


@dataclass(slots=True)
class CompactionPreparation:
    """Data passed to a compaction hook before summarization."""

    first_kept_entry_id: str
    messages_to_summarize: list[Any]
    tokens_before: int
    previous_summary: str | None = None
    custom_instructions: str | None = None


@dataclass(slots=True)
class CompactionResult:
    """Value returned by a compaction hook to override default summarization."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: Any = None
