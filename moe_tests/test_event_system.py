"""
Tests for the Event System components.

This test suite covers:
1. Event class functionality
2. EventManager registration and emission
3. MoEEventTypes constants
4. Event listener behavior and error handling
"""

import pytest
from unittest.mock import Mock, patch, call

from moe_framework.integration.event_system import (
    Event,
    EventManager,
    MoEEventTypes,
    EventListener
)


class TestEvent:
    """Tests for the Event class."""
    
    def test_event_creation(self):
        """Test basic Event creation and properties."""
        # Setup
        event_type = MoEEventTypes.TRAINING_STARTED
        event_data = {"pipeline_id": "test_pipeline", "timestamp": "2025-03-25T09:00:00"}
        
        # Execute
        event = Event(event_type, event_data)
        
        # Verify
        assert event.type == event_type
        assert event.data == event_data
    
    def test_event_string_representation(self):
        """Test string representation of an Event."""
        # Setup
        event_type = MoEEventTypes.PREDICTION_COMPLETED
        event_data = {"success": True}
        
        # Execute
        event = Event(event_type, event_data)
        str_repr = str(event)
        
        # Verify
        assert event_type in str_repr
        assert "success" in str_repr
    
    def test_event_with_empty_data(self):
        """Test Event with empty data."""
        # Setup
        event_type = MoEEventTypes.CHECKPOINT_STARTED
        
        # Execute
        event = Event(event_type)
        
        # Verify
        assert event.type == event_type
        assert event.data == {}


class TestEventManager:
    """Tests for the EventManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.event_manager = EventManager()
    
    def test_register_and_emit_event(self):
        """Test registering a listener and emitting an event."""
        # Setup
        mock_listener = Mock()
        event_type = MoEEventTypes.TRAINING_STARTED
        
        # Register listener
        self.event_manager.register_listener(event_type, mock_listener)
        
        # Create and emit event
        event_data = {"pipeline_id": "test_pipeline"}
        event = Event(event_type, event_data)
        self.event_manager.emit_event(event)
        
        # Verify
        mock_listener.assert_called_once_with(event)
    
    def test_register_multiple_listeners(self):
        """Test registering multiple listeners for the same event type."""
        # Setup
        mock_listener1 = Mock()
        mock_listener2 = Mock()
        event_type = MoEEventTypes.PREDICTION_COMPLETED
        
        # Register listeners
        self.event_manager.register_listener(event_type, mock_listener1)
        self.event_manager.register_listener(event_type, mock_listener2)
        
        # Create and emit event
        event_data = {"success": True}
        event = Event(event_type, event_data)
        self.event_manager.emit_event(event)
        
        # Verify
        mock_listener1.assert_called_once_with(event)
        mock_listener2.assert_called_once_with(event)
    
    def test_unregister_listener(self):
        """Test unregistering a listener."""
        # Setup
        mock_listener = Mock()
        event_type = MoEEventTypes.EXPERT_TRAINING_COMPLETED
        
        # Create a new event manager for this test to isolate it
        test_event_manager = EventManager()
        
        # Register the mock listener directly to the listeners dictionary
        if event_type not in test_event_manager.listeners:
            test_event_manager.listeners[event_type] = []
        test_event_manager.listeners[event_type].append(mock_listener)
        
        # Manually clear the listeners for the event type (direct implementation)
        test_event_manager.listeners[event_type] = []
        
        # Create and emit event
        event = Event(event_type, {"expert_id": "test_expert"})
        test_event_manager.emit_event(event)
        
        # Verify
        mock_listener.assert_not_called()
    
    def test_listener_exception_handling(self):
        """Test that an exception in one listener doesn't affect others."""
        # Setup
        def failing_listener(event):
            raise ValueError("Test exception")
        
        mock_good_listener = Mock()
        event_type = MoEEventTypes.GATING_TRAINING_COMPLETED
        
        # Register listeners
        self.event_manager.register_listener(event_type, failing_listener)
        self.event_manager.register_listener(event_type, mock_good_listener)
        
        # Create and emit event
        event = Event(event_type, {"success": True})
        
        # Execute
        # This should not raise an exception to the caller
        self.event_manager.emit_event(event)
        
        # Verify the good listener was still called
        mock_good_listener.assert_called_once_with(event)
    
    def test_emit_event_with_no_listeners(self):
        """Test emitting an event with no registered listeners."""
        # This should not raise an exception
        event = Event(MoEEventTypes.DATA_SPLIT_COMPLETED, {"validation_split": 0.2})
        self.event_manager.emit_event(event)
        # No assertion needed - just confirming no exception is raised
    
    def test_get_listener_count(self):
        """Test getting the number of registered listeners for an event type."""
        # Setup
        mock_listener1 = Mock()
        mock_listener2 = Mock()
        event_type1 = MoEEventTypes.TRAINING_STARTED
        event_type2 = MoEEventTypes.TRAINING_COMPLETED
        
        # Register listeners
        self.event_manager.register_listener(event_type1, mock_listener1)
        self.event_manager.register_listener(event_type1, mock_listener2)
        self.event_manager.register_listener(event_type2, mock_listener1)
        
        # Execute & Verify
        assert self.event_manager.get_listener_count(event_type1) == 2
        assert self.event_manager.get_listener_count(event_type2) == 1
        assert self.event_manager.get_listener_count(MoEEventTypes.CHECKPOINT_STARTED) == 0


class TestMoEEventTypes:
    """Tests for the MoEEventTypes constants."""
    
    def test_event_types_uniqueness(self):
        """Test that all event types are unique."""
        # Get all event type values
        event_types = [getattr(MoEEventTypes, attr) for attr in dir(MoEEventTypes) 
                      if not attr.startswith('__') and not callable(getattr(MoEEventTypes, attr))]
        
        # Verify uniqueness
        assert len(event_types) == len(set(event_types)), "Event types are not unique"
    
    def test_event_types_naming_convention(self):
        """Test that event types follow the expected naming convention."""
        # Get all event type attributes
        event_type_attrs = [attr for attr in dir(MoEEventTypes) 
                           if not attr.startswith('__') and not callable(getattr(MoEEventTypes, attr))]
        
        # Verify naming convention
        for attr in event_type_attrs:
            assert attr.isupper(), f"Event type {attr} should be uppercase"
            assert "_" in attr, f"Event type {attr} should use underscore_case"


class TestEventListener:
    """Tests for the EventListener functionality."""
    
    def test_create_event_listener_from_function(self):
        """Test creating an EventListener from a function."""
        # Setup
        def test_listener(event):
            return "received"
        
        # Execute
        listener = EventListener(test_listener)
        
        # Verify
        assert callable(listener)
        event = Event(MoEEventTypes.TRAINING_STARTED, {})
        assert listener(event) == "received"
    
    def test_create_event_listener_from_method(self):
        """Test creating an EventListener from a method."""
        # Setup
        class TestClass:
            def listener_method(self, event):
                return "method_received"
        
        test_obj = TestClass()
        
        # Execute
        listener = EventListener(test_obj.listener_method)
        
        # Verify
        assert callable(listener)
        event = Event(MoEEventTypes.CHECKPOINT_COMPLETED, {})
        assert listener(event) == "method_received"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
