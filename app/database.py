import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required. Set it to your Supabase PostgreSQL connection string."
    )

# Supabase may provide a URL without an explicit SQLAlchemy driver.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def init_db():
    """Create chat tables if they do not exist."""
    Base.metadata.create_all(engine)
    logger.info("PostgreSQL chat tables are ready")


def create_chat_session() -> str:
    session_id = str(uuid4())
    with Session(engine) as db:
        db.add(ChatSession(id=session_id))
        db.commit()
    return session_id


def chat_session_exists(session_id: str) -> bool:
    with Session(engine) as db:
        return db.get(ChatSession, session_id) is not None


def save_message(session_id: str, role: str, content: str):
    with Session(engine) as db:
        db.add(ChatMessage(session_id=session_id, role=role, content=content))
        db.commit()


def get_messages(session_id: str) -> list[dict]:
    with Session(engine) as db:
        messages = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        ).all()
        

    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]


def list_chat_sessions() -> list[dict]:
    with Session(engine) as db:
        sessions = db.scalars(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        ).all()
        result = []
        for chat_session in sessions:
            latest_message = db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == chat_session.id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(1)
            ).first()
            result.append({
                "session_id": chat_session.id,
                "preview": latest_message.content if latest_message else "New chat",
                "created_at": chat_session.created_at.isoformat(),
            })
        return result
