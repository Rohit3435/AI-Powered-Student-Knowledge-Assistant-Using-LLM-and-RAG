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


def test_main_accepts_password_through_normal_terminal_input(monkeypatch, capsys):
    user_inputs = iter(["s", "riya.mehta@nsut.ac.in", "secret", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(user_inputs))
    monkeypatch.setattr(main, "is_student_email", lambda _email: True)
    monkeypatch.setattr(main, "authenticate_student", lambda _email, password: password == "secret")
    monkeypatch.setattr(main, "student_name_for_email", lambda _email: "Riya Mehta")
    monkeypatch.setattr(main, "prepare_pipeline", lambda **_kwargs: {
        "collection": object(), "embedding_model": object(), "documents": [],
    })

    main.main()

    output = capsys.readouterr().out
    assert "Loaded 0 document file(s)" in output
    assert "Exiting" in output
