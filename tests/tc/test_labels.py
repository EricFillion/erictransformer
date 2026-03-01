from tests.tc import (
    PROVIDED_LABELS,
    eric_tc_provided_labels,
)


def test_provided_labels():
    result = eric_tc_provided_labels("hi")

    for label in PROVIDED_LABELS:
        assert label in result.labels

    assert len(result.labels) == len(PROVIDED_LABELS)

    assert result.labels[0] in PROVIDED_LABELS
