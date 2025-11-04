"""FastAPI routes for Genetic Algorithm project."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from .genetic_algorithm import GAConfig
from .session_manager import get_session_manager

router = APIRouter(prefix="/projectGA", tags=["projectGA"])


# Pydantic models for request/response
class InitializeRequest(BaseModel):
    """Request to initialize GA."""
    target: str
    population_size: int = 200
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    elitism_count: int = 2
    tournament_size: int = 5
    charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "


class EvolveRequest(BaseModel):
    """Request to evolve one generation."""
    generations: int = 1  # Number of generations to evolve


class Individual(BaseModel):
    """Individual from population."""
    genes: str
    fitness: float


class StatusResponse(BaseModel):
    """Response with GA status."""
    session_id: str
    initialized: bool
    generation: int
    population_size: int
    target: str
    best_fitness: float
    best_individual: str
    average_fitness: float
    is_complete: bool
    config: dict


class EvolutionResponse(BaseModel):
    """Response from evolution step."""
    session_id: str
    generation: int
    best_fitness: float
    best_individual: str
    average_fitness: float
    is_complete: bool
    generations_evolved: int


class PopulationResponse(BaseModel):
    """Response with population data."""
    session_id: str
    generation: int
    top_individuals: List[Individual]
    average_fitness: float
    best_fitness: float


class HistoryDataPoint(BaseModel):
    """Single data point in evolution history."""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    best_individual: str


class HistoryResponse(BaseModel):
    """Response with evolution history."""
    session_id: str
    target: str
    history: List[HistoryDataPoint]


# Shakespeare quotes for easy selection
SHAKESPEARE_QUOTES = [
    "TO BE OR NOT TO BE",
    "ALL THE WORLDS A STAGE",
    "LOVE ALL TRUST A FEW",
    "BREVITY IS THE SOUL OF WIT",
    "THE COURSE OF TRUE LOVE",
    "WHAT A PIECE OF WORK IS MAN",
    "COWARDS DIE MANY TIMES",
    "SOME ARE BORN GREAT",
    "TO ERR IS HUMAN",
    "PATIENCE IS A VIRTUE"
]


@router.post("/initialize", response_model=StatusResponse)
def initialize(request: InitializeRequest) -> StatusResponse:
    """Initialize a new GA session."""
    try:
        session_manager = get_session_manager()

        # Create configuration
        config = GAConfig(
            population_size=request.population_size,
            mutation_rate=request.mutation_rate,
            crossover_rate=request.crossover_rate,
            elitism_count=request.elitism_count,
            tournament_size=request.tournament_size,
            charset=request.charset
        )

        # Create new session
        session_id = session_manager.create_session(config)
        session = session_manager.get_session(session_id)

        if not session:
            raise RuntimeError("Failed to create session")

        # Initialize GA with target
        session.ga.initialize(request.target)
        session.is_active = True

        status = session.ga.get_status()
        return StatusResponse(
            session_id=session_id,
            **status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve/{session_id}", response_model=EvolutionResponse)
def evolve(
    session_id: str,
    request: EvolveRequest
) -> EvolutionResponse:
    """Evolve the population for N generations."""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if not session.ga.population:
            raise HTTPException(
                status_code=400,
                detail="GA not initialized. Call /initialize first."
            )

        # Evolve for requested generations
        generations_evolved = 0

        for _ in range(request.generations):
            is_complete = session.ga.evolve()
            generations_evolved += 1

            if is_complete:
                break

        status = session.ga.get_status()
        return EvolutionResponse(
            session_id=session_id,
            generation=status["generation"],
            best_fitness=status["best_fitness"],
            best_individual=status["best_individual"],
            average_fitness=status["average_fitness"],
            is_complete=status["is_complete"],
            generations_evolved=generations_evolved
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=StatusResponse)
def get_status(session_id: str) -> StatusResponse:
    """Get current GA status."""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        status = session.ga.get_status()
        return StatusResponse(
            session_id=session_id,
            **status
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/population/{session_id}", response_model=PopulationResponse)
def get_population(
    session_id: str,
    top_n: int = Query(20, ge=1, le=100)
) -> PopulationResponse:
    """Get top individuals from population."""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if not session.ga.population:
            raise HTTPException(
                status_code=400,
                detail="No population data available"
            )

        top_individuals = session.ga.get_top_population(top_n)
        status = session.ga.get_status()

        return PopulationResponse(
            session_id=session_id,
            generation=status["generation"],
            top_individuals=[Individual(**ind) for ind in top_individuals],
            average_fitness=status["average_fitness"],
            best_fitness=status["best_fitness"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """Get evolution history."""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        history = session.ga.get_history()
        status = session.ga.get_status()

        return HistoryResponse(
            session_id=session_id,
            target=status["target"],
            history=[HistoryDataPoint(**data) for data in history]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{session_id}")
def reset(session_id: str) -> dict:
    """Reset a GA session."""
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.ga.reset()
        session.is_active = False

        return {
            "success": True,
            "message": "Session reset successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a GA session."""
    try:
        session_manager = get_session_manager()

        if session_manager.delete_session(session_id):
            return {
                "success": True,
                "message": "Session deleted successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotes")
def get_quotes() -> dict:
    """Get list of Shakespeare quotes."""
    return {
        "quotes": SHAKESPEARE_QUOTES
    }
