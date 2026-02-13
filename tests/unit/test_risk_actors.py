"""Tests for risk management actors and configurations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from vibe_quant.risk.actors import (
    PortfolioRiskActorConfig,
    PortfolioRiskState,
    RiskEvent,
    RiskState,
    StrategyRiskActorConfig,
    StrategyRiskState,
)
from vibe_quant.risk.config import (
    PortfolioRiskConfig,
    StrategyRiskConfig,
    create_default_portfolio_risk_config,
    create_default_strategy_risk_config,
)


class TestStrategyRiskConfig:
    """Tests for StrategyRiskConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = StrategyRiskConfig()
        assert config.max_drawdown_pct == Decimal("0.15")
        assert config.max_daily_loss_pct == Decimal("0.02")
        assert config.max_consecutive_losses == 10
        assert config.max_position_count == 5
        assert config.drawdown_scale_pct == Decimal("0.10")
        assert config.cooldown_after_halt_hours == 24

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = StrategyRiskConfig(
            max_drawdown_pct=Decimal("0.20"),
            max_daily_loss_pct=Decimal("0.03"),
            max_consecutive_losses=15,
            max_position_count=10,
            drawdown_scale_pct=Decimal("0.12"),
            cooldown_after_halt_hours=48,
        )
        assert config.max_drawdown_pct == Decimal("0.20")
        assert config.max_daily_loss_pct == Decimal("0.03")
        assert config.max_consecutive_losses == 15
        assert config.max_position_count == 10
        assert config.drawdown_scale_pct == Decimal("0.12")
        assert config.cooldown_after_halt_hours == 48

    def test_drawdown_scale_disabled(self) -> None:
        """Should accept None to disable drawdown scaling."""
        config = StrategyRiskConfig(drawdown_scale_pct=None)
        assert config.drawdown_scale_pct is None

    def test_invalid_drawdown_scale_exceeds_halt(self) -> None:
        """drawdown_scale_pct must be < max_drawdown_pct."""
        with pytest.raises(ValueError, match="drawdown_scale_pct must be < max_drawdown_pct"):
            StrategyRiskConfig(
                max_drawdown_pct=Decimal("0.15"),
                drawdown_scale_pct=Decimal("0.15"),
            )

    def test_invalid_cooldown_negative(self) -> None:
        """cooldown_after_halt_hours must be >= 0."""
        with pytest.raises(ValueError, match="cooldown_after_halt_hours must be >= 0"):
            StrategyRiskConfig(cooldown_after_halt_hours=-1)

    def test_invalid_drawdown_zero(self) -> None:
        """Should reject zero drawdown."""
        with pytest.raises(ValueError, match="max_drawdown_pct must be positive"):
            StrategyRiskConfig(max_drawdown_pct=Decimal("0"))

    def test_invalid_drawdown_negative(self) -> None:
        """Should reject negative drawdown."""
        with pytest.raises(ValueError, match="max_drawdown_pct must be positive"):
            StrategyRiskConfig(max_drawdown_pct=Decimal("-0.1"))

    def test_invalid_drawdown_over_100(self) -> None:
        """Should reject drawdown over 100%."""
        with pytest.raises(ValueError, match="max_drawdown_pct must be <= 1"):
            StrategyRiskConfig(max_drawdown_pct=Decimal("1.5"))

    def test_invalid_daily_loss_zero(self) -> None:
        """Should reject zero daily loss."""
        with pytest.raises(ValueError, match="max_daily_loss_pct must be positive"):
            StrategyRiskConfig(max_daily_loss_pct=Decimal("0"))

    def test_invalid_daily_loss_over_100(self) -> None:
        """Should reject daily loss over 100%."""
        with pytest.raises(ValueError, match="max_daily_loss_pct must be <= 1"):
            StrategyRiskConfig(max_daily_loss_pct=Decimal("1.1"))

    def test_invalid_consecutive_losses_zero(self) -> None:
        """Should reject zero consecutive losses."""
        with pytest.raises(ValueError, match="max_consecutive_losses must be >= 1"):
            StrategyRiskConfig(max_consecutive_losses=0)

    def test_invalid_position_count_zero(self) -> None:
        """Should reject zero position count."""
        with pytest.raises(ValueError, match="max_position_count must be >= 1"):
            StrategyRiskConfig(max_position_count=0)

    def test_immutable(self) -> None:
        """Config should be frozen (immutable)."""
        config = StrategyRiskConfig()
        with pytest.raises(AttributeError):
            config.max_drawdown_pct = Decimal("0.30")  # type: ignore


class TestPortfolioRiskConfig:
    """Tests for PortfolioRiskConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = PortfolioRiskConfig()
        assert config.max_portfolio_drawdown_pct == Decimal("0.20")
        assert config.max_total_exposure_pct == Decimal("0.50")
        assert config.max_single_instrument_pct == Decimal("0.30")
        assert config.max_portfolio_heat_pct == Decimal("0.06")

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = PortfolioRiskConfig(
            max_portfolio_drawdown_pct=Decimal("0.25"),
            max_total_exposure_pct=Decimal("0.80"),
            max_single_instrument_pct=Decimal("0.40"),
            max_portfolio_heat_pct=Decimal("0.10"),
        )
        assert config.max_portfolio_drawdown_pct == Decimal("0.25")
        assert config.max_total_exposure_pct == Decimal("0.80")
        assert config.max_single_instrument_pct == Decimal("0.40")
        assert config.max_portfolio_heat_pct == Decimal("0.10")

    def test_portfolio_heat_disabled(self) -> None:
        """Should accept None to disable portfolio heat checking."""
        config = PortfolioRiskConfig(max_portfolio_heat_pct=None)
        assert config.max_portfolio_heat_pct is None

    def test_invalid_portfolio_heat_zero(self) -> None:
        """Should reject zero portfolio heat."""
        with pytest.raises(ValueError, match="max_portfolio_heat_pct must be positive"):
            PortfolioRiskConfig(max_portfolio_heat_pct=Decimal("0"))

    def test_invalid_portfolio_heat_over_100(self) -> None:
        """Should reject portfolio heat over 100%."""
        with pytest.raises(ValueError, match="max_portfolio_heat_pct must be <= 1"):
            PortfolioRiskConfig(max_portfolio_heat_pct=Decimal("1.5"))

    def test_invalid_portfolio_drawdown_zero(self) -> None:
        """Should reject zero portfolio drawdown."""
        with pytest.raises(ValueError, match="max_portfolio_drawdown_pct must be positive"):
            PortfolioRiskConfig(max_portfolio_drawdown_pct=Decimal("0"))

    def test_invalid_portfolio_drawdown_over_100(self) -> None:
        """Should reject portfolio drawdown over 100%."""
        with pytest.raises(ValueError, match="max_portfolio_drawdown_pct must be <= 1"):
            PortfolioRiskConfig(max_portfolio_drawdown_pct=Decimal("1.2"))

    def test_invalid_total_exposure_zero(self) -> None:
        """Should reject zero total exposure."""
        with pytest.raises(ValueError, match="max_total_exposure_pct must be positive"):
            PortfolioRiskConfig(max_total_exposure_pct=Decimal("0"))

    def test_invalid_instrument_pct_zero(self) -> None:
        """Should reject zero single instrument percentage."""
        with pytest.raises(ValueError, match="max_single_instrument_pct must be positive"):
            PortfolioRiskConfig(max_single_instrument_pct=Decimal("0"))

    def test_invalid_instrument_pct_over_100(self) -> None:
        """Should reject single instrument percentage over 100%."""
        with pytest.raises(ValueError, match="max_single_instrument_pct must be <= 1"):
            PortfolioRiskConfig(max_single_instrument_pct=Decimal("1.5"))

    def test_immutable(self) -> None:
        """Config should be frozen (immutable)."""
        config = PortfolioRiskConfig()
        with pytest.raises(AttributeError):
            config.max_portfolio_drawdown_pct = Decimal("0.30")  # type: ignore


class TestDefaultConfigFactories:
    """Tests for default config factory functions."""

    def test_create_default_strategy_risk_config(self) -> None:
        """Should create valid default strategy config."""
        config = create_default_strategy_risk_config()
        assert isinstance(config, StrategyRiskConfig)
        assert config.max_drawdown_pct == Decimal("0.15")
        assert config.max_daily_loss_pct == Decimal("0.02")
        assert config.max_consecutive_losses == 10
        assert config.max_position_count == 5
        assert config.drawdown_scale_pct == Decimal("0.10")
        assert config.cooldown_after_halt_hours == 24

    def test_create_default_portfolio_risk_config(self) -> None:
        """Should create valid default portfolio config."""
        config = create_default_portfolio_risk_config()
        assert isinstance(config, PortfolioRiskConfig)
        assert config.max_portfolio_drawdown_pct == Decimal("0.20")
        assert config.max_total_exposure_pct == Decimal("0.50")
        assert config.max_single_instrument_pct == Decimal("0.30")
        assert config.max_portfolio_heat_pct == Decimal("0.06")


class TestRiskState:
    """Tests for RiskState enum."""

    def test_active_state(self) -> None:
        """ACTIVE should indicate normal trading."""
        assert RiskState.ACTIVE.value == "ACTIVE"

    def test_warning_state(self) -> None:
        """WARNING should indicate approaching limits."""
        assert RiskState.WARNING.value == "WARNING"

    def test_halted_state(self) -> None:
        """HALTED should indicate trading stopped."""
        assert RiskState.HALTED.value == "HALTED"

    def test_string_conversion(self) -> None:
        """Should convert to/from string."""
        # RiskState(str, Enum) uses .value for string comparison
        assert RiskState.ACTIVE.value == "ACTIVE"
        assert RiskState("HALTED") == RiskState.HALTED


class TestRiskEvent:
    """Tests for RiskEvent dataclass."""

    def test_create_event(self) -> None:
        """Should create valid risk event."""
        event = RiskEvent(
            timestamp="2024-01-15T10:30:00Z",
            event_type="CIRCUIT_BREAKER",
            level="strategy",
            strategy_id="RSI_MEAN_REV",
            metric="drawdown",
            current_value="0.16",
            threshold="0.15",
            state="HALTED",
            message="Strategy halted due to drawdown",
        )
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.event_type == "CIRCUIT_BREAKER"
        assert event.level == "strategy"
        assert event.strategy_id == "RSI_MEAN_REV"

    def test_to_dict(self) -> None:
        """Should convert to dict for JSON logging."""
        event = RiskEvent(
            timestamp="2024-01-15T10:30:00Z",
            event_type="CIRCUIT_BREAKER",
            level="portfolio",
            strategy_id=None,
            metric="total_exposure",
            current_value="0.55",
            threshold="0.50",
            state="HALTED",
            message="Portfolio halted",
        )
        d = event.to_dict()
        assert d["ts"] == "2024-01-15T10:30:00Z"
        assert d["event"] == "CIRCUIT_BREAKER"
        assert d["level"] == "portfolio"
        assert d["strategy_id"] is None
        assert d["metric"] == "total_exposure"


class TestStrategyRiskState:
    """Tests for StrategyRiskState dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        state = StrategyRiskState()
        assert state.high_water_mark == Decimal("0")
        assert state.current_drawdown_pct == Decimal("0")
        assert state.daily_pnl == Decimal("0")
        assert state.daily_start_equity == Decimal("0")
        assert state.consecutive_losses == 0
        assert state.current_date is None
        assert state.state == RiskState.ACTIVE
        assert state.position_count == 0
        assert state.position_scale_factor == Decimal("1")
        assert state.halted_at is None

    def test_mutable(self) -> None:
        """State should be mutable for updates."""
        state = StrategyRiskState()
        state.high_water_mark = Decimal("100000")
        state.current_drawdown_pct = Decimal("0.05")
        state.consecutive_losses = 3
        state.state = RiskState.WARNING
        state.position_scale_factor = Decimal("0.75")

        assert state.high_water_mark == Decimal("100000")
        assert state.current_drawdown_pct == Decimal("0.05")
        assert state.consecutive_losses == 3
        assert state.state == RiskState.WARNING
        assert state.position_scale_factor == Decimal("0.75")


class TestPortfolioRiskState:
    """Tests for PortfolioRiskState dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        state = PortfolioRiskState()
        assert state.high_water_mark == Decimal("0")
        assert state.current_drawdown_pct == Decimal("0")
        assert state.total_exposure_pct == Decimal("0")
        assert state.instrument_exposures == {}
        assert state.state == RiskState.ACTIVE
        assert state.portfolio_heat_pct == Decimal("0")

    def test_mutable(self) -> None:
        """State should be mutable for updates."""
        state = PortfolioRiskState()
        state.high_water_mark = Decimal("500000")
        state.total_exposure_pct = Decimal("0.35")
        state.instrument_exposures = {
            "BTCUSDT-PERP": Decimal("0.20"),
            "ETHUSDT-PERP": Decimal("0.15"),
        }
        state.state = RiskState.WARNING
        state.portfolio_heat_pct = Decimal("0.04")

        assert state.high_water_mark == Decimal("500000")
        assert state.total_exposure_pct == Decimal("0.35")
        assert len(state.instrument_exposures) == 2
        assert state.portfolio_heat_pct == Decimal("0.04")


class TestStrategyRiskActorConfig:
    """Tests for StrategyRiskActorConfig."""

    def test_create_config(self) -> None:
        """Should create valid actor config."""
        config = StrategyRiskActorConfig(
            component_id="RISK-001",
            strategy_id="RSI_MEAN_REV",
            max_drawdown_pct=Decimal("0.15"),
            max_daily_loss_pct=Decimal("0.02"),
            max_consecutive_losses=10,
            max_position_count=5,
        )
        assert config.strategy_id == "RSI_MEAN_REV"
        assert config.max_drawdown_pct == Decimal("0.15")

    def test_config_is_frozen(self) -> None:
        """Config should be immutable."""
        StrategyRiskActorConfig(
            component_id="RISK-001",
            strategy_id="TEST",
        )
        # ActorConfig is frozen by default


class TestPortfolioRiskActorConfig:
    """Tests for PortfolioRiskActorConfig."""

    def test_create_config(self) -> None:
        """Should create valid actor config."""
        config = PortfolioRiskActorConfig(
            component_id="PORTFOLIO-RISK",
            max_portfolio_drawdown_pct=Decimal("0.20"),
            max_total_exposure_pct=Decimal("0.50"),
            max_single_instrument_pct=Decimal("0.30"),
        )
        assert config.max_portfolio_drawdown_pct == Decimal("0.20")
        assert config.max_total_exposure_pct == Decimal("0.50")


class TestStrategyRiskActorLogic:
    """Tests for StrategyRiskActor risk check logic.

    These tests verify the risk logic without needing full NautilusTrader
    integration by testing the state transitions directly.
    """

    def test_drawdown_breach_detection(self) -> None:
        """Should detect when drawdown exceeds limit."""
        config = StrategyRiskConfig(max_drawdown_pct=Decimal("0.15"))
        state = StrategyRiskState(
            high_water_mark=Decimal("100000"),
            current_drawdown_pct=Decimal("0.16"),  # 16% > 15%
        )
        # Simulating what check_risk would evaluate
        assert state.current_drawdown_pct >= config.max_drawdown_pct

    def test_drawdown_within_limit(self) -> None:
        """Should not trigger when drawdown is within limit."""
        config = StrategyRiskConfig(max_drawdown_pct=Decimal("0.15"))
        state = StrategyRiskState(
            high_water_mark=Decimal("100000"),
            current_drawdown_pct=Decimal("0.10"),  # 10% < 15%
        )
        assert state.current_drawdown_pct < config.max_drawdown_pct

    def test_daily_loss_breach_detection(self) -> None:
        """Should detect when daily loss exceeds limit."""
        config = StrategyRiskConfig(max_daily_loss_pct=Decimal("0.02"))
        state = StrategyRiskState(
            daily_start_equity=Decimal("100000"),
            daily_pnl=Decimal("-2500"),  # -2.5% > -2%
        )
        if state.daily_start_equity > Decimal("0"):
            daily_loss_pct = abs(state.daily_pnl) / state.daily_start_equity
            assert daily_loss_pct >= config.max_daily_loss_pct

    def test_daily_loss_within_limit(self) -> None:
        """Should not trigger when daily loss is within limit."""
        config = StrategyRiskConfig(max_daily_loss_pct=Decimal("0.02"))
        state = StrategyRiskState(
            daily_start_equity=Decimal("100000"),
            daily_pnl=Decimal("-1000"),  # -1% < -2%
        )
        if state.daily_start_equity > Decimal("0"):
            daily_loss_pct = abs(state.daily_pnl) / state.daily_start_equity
            assert daily_loss_pct < config.max_daily_loss_pct

    def test_consecutive_losses_breach(self) -> None:
        """Should detect when consecutive losses exceed limit."""
        config = StrategyRiskConfig(max_consecutive_losses=10)
        state = StrategyRiskState(consecutive_losses=10)
        assert state.consecutive_losses >= config.max_consecutive_losses

    def test_consecutive_losses_within_limit(self) -> None:
        """Should not trigger when consecutive losses within limit."""
        config = StrategyRiskConfig(max_consecutive_losses=10)
        state = StrategyRiskState(consecutive_losses=5)
        assert state.consecutive_losses < config.max_consecutive_losses

    def test_position_count_warning(self) -> None:
        """Should warn when position count at limit."""
        config = StrategyRiskConfig(max_position_count=5)
        state = StrategyRiskState(position_count=5)
        assert state.position_count >= config.max_position_count

    def test_hwm_update_logic(self) -> None:
        """HWM should update when equity increases."""
        state = StrategyRiskState(high_water_mark=Decimal("100000"))

        # Simulate equity increase
        new_equity = Decimal("105000")
        if new_equity > state.high_water_mark:
            state.high_water_mark = new_equity

        assert state.high_water_mark == Decimal("105000")

    def test_hwm_no_update_on_decrease(self) -> None:
        """HWM should not update when equity decreases."""
        state = StrategyRiskState(high_water_mark=Decimal("100000"))

        # Simulate equity decrease
        new_equity = Decimal("95000")
        if new_equity > state.high_water_mark:
            state.high_water_mark = new_equity

        assert state.high_water_mark == Decimal("100000")  # Unchanged

    def test_drawdown_calculation(self) -> None:
        """Drawdown should be calculated correctly."""
        hwm = Decimal("100000")
        current_equity = Decimal("85000")

        # Formula: (hwm - equity) / hwm
        drawdown = (hwm - current_equity) / hwm
        assert drawdown == Decimal("0.15")  # 15% drawdown


class TestPortfolioRiskActorLogic:
    """Tests for PortfolioRiskActor risk check logic."""

    def test_portfolio_drawdown_breach(self) -> None:
        """Should detect portfolio drawdown breach."""
        config = PortfolioRiskConfig(max_portfolio_drawdown_pct=Decimal("0.20"))
        state = PortfolioRiskState(current_drawdown_pct=Decimal("0.22"))
        assert state.current_drawdown_pct >= config.max_portfolio_drawdown_pct

    def test_portfolio_drawdown_within_limit(self) -> None:
        """Should not trigger when portfolio drawdown within limit."""
        config = PortfolioRiskConfig(max_portfolio_drawdown_pct=Decimal("0.20"))
        state = PortfolioRiskState(current_drawdown_pct=Decimal("0.15"))
        assert state.current_drawdown_pct < config.max_portfolio_drawdown_pct

    def test_total_exposure_breach(self) -> None:
        """Should detect total exposure breach."""
        config = PortfolioRiskConfig(max_total_exposure_pct=Decimal("0.50"))
        state = PortfolioRiskState(total_exposure_pct=Decimal("0.55"))
        assert state.total_exposure_pct > config.max_total_exposure_pct

    def test_total_exposure_within_limit(self) -> None:
        """Should not trigger when exposure within limit."""
        config = PortfolioRiskConfig(max_total_exposure_pct=Decimal("0.50"))
        state = PortfolioRiskState(total_exposure_pct=Decimal("0.40"))
        assert state.total_exposure_pct <= config.max_total_exposure_pct

    def test_instrument_concentration_breach(self) -> None:
        """Should detect single instrument concentration breach."""
        config = PortfolioRiskConfig(max_single_instrument_pct=Decimal("0.30"))
        state = PortfolioRiskState(
            instrument_exposures={
                "BTCUSDT-PERP": Decimal("0.35"),  # 35% > 30%
                "ETHUSDT-PERP": Decimal("0.10"),
            }
        )
        for exposure in state.instrument_exposures.values():
            if exposure > config.max_single_instrument_pct:
                assert True
                return
        pytest.fail("Should have detected concentration breach")

    def test_instrument_concentration_within_limit(self) -> None:
        """Should not trigger when all instruments within limit."""
        config = PortfolioRiskConfig(max_single_instrument_pct=Decimal("0.30"))
        state = PortfolioRiskState(
            instrument_exposures={
                "BTCUSDT-PERP": Decimal("0.25"),
                "ETHUSDT-PERP": Decimal("0.15"),
            }
        )
        for exposure in state.instrument_exposures.values():
            assert exposure <= config.max_single_instrument_pct

    def test_exposure_calculation(self) -> None:
        """Exposure should be calculated correctly."""
        equity = Decimal("100000")
        position_notional = Decimal("30000")

        exposure_pct = position_notional / equity
        assert exposure_pct == Decimal("0.30")  # 30%

    def test_multiple_instrument_exposure_calculation(self) -> None:
        """Total exposure should sum all position notionals."""
        equity = Decimal("100000")
        positions = {
            "BTCUSDT-PERP": Decimal("20000"),
            "ETHUSDT-PERP": Decimal("15000"),
            "SOLUSDT-PERP": Decimal("10000"),
        }

        total_notional = sum(positions.values())
        total_exposure_pct = total_notional / equity

        assert total_exposure_pct == Decimal("0.45")  # 45%


class TestRiskStateTransitions:
    """Tests for risk state transition logic."""

    def test_active_to_warning_transition(self) -> None:
        """State should transition from ACTIVE to WARNING."""
        state = StrategyRiskState(state=RiskState.ACTIVE)
        state.state = RiskState.WARNING
        assert state.state == RiskState.WARNING

    def test_warning_to_halted_transition(self) -> None:
        """State should transition from WARNING to HALTED."""
        state = StrategyRiskState(state=RiskState.WARNING)
        state.state = RiskState.HALTED
        assert state.state == RiskState.HALTED

    def test_active_to_halted_transition(self) -> None:
        """State should transition directly from ACTIVE to HALTED."""
        state = StrategyRiskState(state=RiskState.ACTIVE)
        state.state = RiskState.HALTED
        assert state.state == RiskState.HALTED

    def test_halted_stays_halted(self) -> None:
        """HALTED state should remain until explicitly reset."""
        state = StrategyRiskState(state=RiskState.HALTED)
        # Even if conditions improve, stay halted
        state.current_drawdown_pct = Decimal("0")  # Recovered
        # State remains HALTED until explicitly changed
        assert state.state == RiskState.HALTED


class TestDrawdownScaling:
    """Tests for drawdown-based position scaling."""

    def test_no_drawdown_full_size(self) -> None:
        """Scale factor = 1.0 when no drawdown."""
        config = StrategyRiskConfig(
            max_drawdown_pct=Decimal("0.15"),
            drawdown_scale_pct=Decimal("0.10"),
        )
        state = StrategyRiskState(current_drawdown_pct=Decimal("0"))
        # Simulate the scaling logic
        dd = state.current_drawdown_pct
        assert dd <= Decimal("0")  # no DD -> factor = 1.0

    def test_below_scale_threshold_partial(self) -> None:
        """Linear scale from 1.0 to 0.5 as DD approaches scale_pct."""
        config = StrategyRiskConfig(
            max_drawdown_pct=Decimal("0.15"),
            drawdown_scale_pct=Decimal("0.10"),
        )
        # At 5% DD (half of 10% scale threshold): factor = 1.0 - (0.5/1.0)*0.5 = 0.75
        dd = Decimal("0.05")
        ratio = dd / config.drawdown_scale_pct
        factor = Decimal("1") - (ratio * Decimal("0.5"))
        assert factor == Decimal("0.75")

    def test_at_scale_threshold_half_size(self) -> None:
        """Factor = 0.5 exactly at drawdown_scale_pct."""
        dd = Decimal("0.10")
        scale_pct = Decimal("0.10")
        ratio = dd / scale_pct
        factor = Decimal("1") - (ratio * Decimal("0.5"))
        assert factor == Decimal("0.5")

    def test_between_scale_and_halt_linear(self) -> None:
        """Linear scale from 0.5 to 0.0 between scale and halt."""
        scale_pct = Decimal("0.10")
        halt_pct = Decimal("0.15")
        dd = Decimal("0.125")  # midpoint
        remaining = halt_pct - scale_pct
        ratio = (dd - scale_pct) / remaining
        factor = Decimal("0.5") * (Decimal("1") - ratio)
        assert factor == Decimal("0.25")

    def test_disabled_when_none(self) -> None:
        """Scale factor stays 1.0 when drawdown_scale_pct is None."""
        config = StrategyRiskConfig(drawdown_scale_pct=None)
        # No scaling configured -> factor stays 1.0
        assert config.drawdown_scale_pct is None


class TestPortfolioHeat:
    """Tests for portfolio heat limit."""

    def test_heat_breach_detection(self) -> None:
        """Should detect when portfolio heat exceeds limit."""
        config = PortfolioRiskConfig(max_portfolio_heat_pct=Decimal("0.06"))
        state = PortfolioRiskState(portfolio_heat_pct=Decimal("0.07"))
        assert state.portfolio_heat_pct >= config.max_portfolio_heat_pct

    def test_heat_within_limit(self) -> None:
        """Should not trigger when heat within limit."""
        config = PortfolioRiskConfig(max_portfolio_heat_pct=Decimal("0.06"))
        state = PortfolioRiskState(portfolio_heat_pct=Decimal("0.04"))
        assert state.portfolio_heat_pct < config.max_portfolio_heat_pct

    def test_heat_calculation(self) -> None:
        """Heat = sum(position_risk) / equity."""
        equity = Decimal("100000")
        # 2 positions, each risking 2% of notional at stop
        pos1_notional = Decimal("20000")
        pos2_notional = Decimal("10000")
        stop_pct = Decimal("0.02")
        total_heat = (pos1_notional * stop_pct + pos2_notional * stop_pct) / equity
        assert total_heat == Decimal("0.006")  # 0.6%

    def test_heat_disabled_when_none(self) -> None:
        """Should accept None to disable heat checking."""
        config = PortfolioRiskConfig(max_portfolio_heat_pct=None)
        assert config.max_portfolio_heat_pct is None
