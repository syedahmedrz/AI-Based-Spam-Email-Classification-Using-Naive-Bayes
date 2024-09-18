import spam_classifier_model as SCM
# from spam_classifier_model import test_data, classify_new_email

# Test with a new email subject
new_email = "Important information about your account"
result = SCM.classify_new_email(new_email)
print(f'The new email is classified as: {result}')

# Test a bunch of subjects
for subject in SCM.test_data:
    test_subject = SCM.classify_new_email(subject)
    print(f'The email \'{subject}\' is classified as: {test_subject}')