"""Tests for the repeat-question terminal session."""

import main


def test_main_reuses_loaded_pipeline_for_multiple_questions(monkeypatch, capsys):
    prepare_calls = []
    answered_questions = []
    answers = iter(["First answer", "Second answer"])
    user_inputs = iter(["g", "Question one", "Question two", "exit"])

    monkeypatch.setattr(main, "prepare_pipeline", lambda **_kwargs: prepare_calls.append(True) or {
        "collection": object(),
        "embedding_model": object(),
        "documents": [],
    })

    def fake_answer(question, _collection, _embedding_model, documents=None, authorized_student_name=None):
        answered_questions.append(question)
        assert documents == []
        assert authorized_student_name is None
        return {"answer": next(answers), "sources": []}

    monkeypatch.setattr(main, "answer_question", fake_answer)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(user_inputs))

    main.main()

    output = capsys.readouterr().out
    assert prepare_calls == [True]
    assert answered_questions == ["Question one", "Question two"]
    assert output.count("Answer:") == 2
    assert "Exiting" in output
