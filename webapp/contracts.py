"""Request bodies shared by the local workbench and multi-user service."""

from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, FiniteFloat, StrictInt


class Detection(BaseModel):
    mode: str = Field(default="auto", pattern="^(auto|tab|notation|both)$")
    source: str = Field(default="auto", pattern="^(auto|image|geometry)$")


class Region(BaseModel):
    page: int = Field(ge=1, le=100)
    kind: Literal["measure", "header", "tempo", "clef", "transposition"]
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    mode: Literal["tab", "notation", "both"] | None = None


class Metadata(BaseModel):
    part_id: str | None = Field(default=None, max_length=160)
    part_name: str | None = Field(default=None, max_length=160)
    title: str = Field(default="未命名乐谱", max_length=500)
    artist: str = Field(default="", max_length=500)
    instrument: Literal["guitar", "bass", "pitched", "drums"] = "guitar"
    midi_program: StrictInt | None = Field(default=None, ge=0, le=127)
    tuning_used: list[Annotated[StrictInt, Field(ge=0, le=127)]] = Field(
        default_factory=lambda: [64, 59, 55, 50, 45, 40], max_length=12
    )
    capo: StrictInt = Field(default=0, ge=0, le=24)
    tempo_quarter: StrictInt = Field(default=120, ge=20, le=400)
    transpose: StrictInt | None = Field(default=None, ge=-36, le=36)


class Boxes(BaseModel):
    boxes: list[Region] = Field(max_length=5000)
    mode: str = Field(default="auto", pattern="^(auto|tab|notation|both)$")


class Correction(BaseModel):
    target: str | None = None
    measure: dict | None = None
    reviewed: bool = False


class Recognition(BaseModel):
    resume: bool = False
    measures: list[Annotated[StrictInt, Field(ge=1)]] | None = Field(
        default=None, min_length=1, max_length=5000
    )


def check_revision(expected: str | None, actual: int):
    if expected is None:
        raise HTTPException(428, "请先读取项目，并在 If-Match 中提供 revision")
    if expected.strip('"') != str(actual):
        raise HTTPException(
            409,
            "其他页面已经修改了这个项目。本页修改尚未保存，请保留需要的内容后重新载入项目。",
        )
