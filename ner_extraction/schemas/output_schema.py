from typing import Any, Iterable  # noqa: N812

from pydantic import Field

from .input_schema import BaseSchema, EntityType, Float, String


class Entity(BaseSchema):
    """A schema for representing named entity recognition output."""

    text: String = Field(description="Text of the entity")
    label: EntityType = Field(description="Label of the entity")  # type: ignore
    score: Float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the entity. Range: 0.0 (least confident) "
        "to 1.0 (most confident)",
    )


class EntitySchemaResponse(BaseSchema):
    """A schema for representing a collection of NER responses with metadata."""

    id: str = Field(description="ID of the input data", alias="txnId")
    text: str = Field(description="The original input data")
    entities: list[Entity | list] = Field(default_factory=list, description="A list of entities")

    def to_sqlalchemy_dict(self) -> dict[str, Any]:
        """Convert to dictionary with SQLAlchemy attribute names."""
        return {
            "txn_id": self.id,
            "text": self.text,
            "entities": [
                entity.model_dump() if hasattr(entity, "model_dump") else entity
                for entity in self.entities
            ],
        }


class AllEntitySchemaResponse(BaseSchema):
    """A schema for representing a collection of entity schema responses."""

    data: list[EntitySchemaResponse]


ResponseGenerator = Iterable[EntitySchemaResponse]
