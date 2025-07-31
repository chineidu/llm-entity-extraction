from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ner_extraction.config import app_config

from .output_schema import EntitySchemaResponse

engine: Engine = create_engine(app_config.database.db_path, echo=False)


class Base(DeclarativeBase):
    pass


class NERData(Base):
    """
    Named Entity Recognition (NER) data model for storing extracted entities.
    """

    __tablename__: str = "ner_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    txn_id: Mapped[str] = mapped_column("txnId", String(50))
    text: Mapped[str] = mapped_column(String(200))
    entities: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    analysis_id: Mapped[str] = mapped_column("analysisId", default="")
    created_at: Mapped[Optional[str]] = mapped_column("createdAt", default=datetime.now)

    def __repr__(self) -> str:
        """
        Returns a string representation of the NERData object.
        """
        return (
            f"User(id={self.id!r}, txn_id={self.txn_id!r}, text={self.text!r}, "
            f"entities={self.entities!r})"
        )


def add_record_to_db(db: Session, data: EntitySchemaResponse | dict[str, Any]) -> NERData:
    """
    Add a new NER record to the database.

    Parameters
    ----------
    db : Session
        SQLAlchemy database session.
    data : EntitySchemaResponse | dict[str, Any]
        Entity schema data containing NER information.

    Returns
    -------
    NERData
        The newly created and persisted NER record.
    """
    if not isinstance(data, EntitySchemaResponse):
        data = EntitySchemaResponse(**data)
    record: NERData = NERData(**data.to_sqlalchemy_dict())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_insert_records(
    db: Session, data_list: list[EntitySchemaResponse] | list[dict[str, Any]]
) -> None:
    """
    High-performance bulk insert using SQLAlchemy Core.

    Parameters
    ----------
    db : Session
        SQLAlchemy database session for handling database operations.
    data_list : list[EntitySchemaResponse] | list[dict[str, Any]]
        List of EntitySchemaResponse objects or dictionaries containing data to be inserted.

    Returns
    -------
    None
        This function performs database operations without returning any value.

    Raises
    ------
    Exception
        Any database-related exception that occurs during bulk insert operation.
    """
    if not data_list:
        return

    if not isinstance(data_list[0], EntitySchemaResponse):
        data_list = [EntitySchemaResponse(**data) for data in data_list]  # type: ignore

    records_data: list[dict[str, Any]] = [data.to_sqlalchemy_dict() for data in data_list]  # type: ignore

    try:
        db.bulk_insert_mappings(NERData, records_data)  # type: ignore
        db.commit()

    except Exception as e:
        db.rollback()
        raise e


def init_db() -> None:
    """
    Initialize the database connection and create all tables.

    Returns
    -------
    None
    """
    # Creates tables
    Base.metadata.create_all(engine)
