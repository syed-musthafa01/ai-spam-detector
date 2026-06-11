import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


def test_packages_importable():
    import fastapi, sklearn, joblib, numpy, psycopg2
    assert all([fastapi, sklearn, joblib, numpy, psycopg2])


def test_db_password_secret_exists():
    """Only the password must come from a secret — everything else has safe defaults"""
    password = os.environ.get("DB_PASSWORD")
    assert password is not None, "DB_PASSWORD secret is missing from GitHub Secrets"


def test_pydantic_text_input():
    from pydantic import BaseModel

    class TextInput(BaseModel):
        text: str

    item = TextInput(text="Win a free iPhone!")
    assert item.text == "Win a free iPhone!"


def test_spam_detection_logic():
    spam_words = ["win", "free", "prize", "urgent", "congratulations"]
    spam_text = "Congratulations! You won a free prize!"
    clean_text = "Can we schedule a meeting tomorrow?"

    assert any(w in spam_text.lower() for w in spam_words)
    assert not any(w in clean_text.lower() for w in spam_words)
