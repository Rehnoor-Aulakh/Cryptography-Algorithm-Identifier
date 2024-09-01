import csv
import io

def text_to_csv_file(text, filename, delimiter=' '):
    # Create an in-memory text stream to hold the CSV data temporarily
    output = io.StringIO()
    
    # Create a CSV writer object associated with the in-memory text stream
    writer = csv.writer(output)
    
    # Split the input text into lines
    lines = text.strip().split('\n')
    
    # Iterate over each line in the text
    for line in lines:
        # Split each line into fields based on the specified delimiter
        fields = line.strip().split(delimiter)
        
        # Write each field on a new line in the CSV file
        for field in fields:
            writer.writerow([field])
    
    # Retrieve the CSV data as a string from the in-memory text stream
    csv_string = output.getvalue()
    
    # Open the specified file in write mode
    with open(filename, 'w', newline='') as file:
        # Write the CSV string to the file
        file.write(csv_string)
    
    # Close the in-memory text stream
    output.close()
    
    
    
text_input=""

output_filename='cipher.csv'

text_to_csv_file(text_input, output_filename)
print(f'CSV data has been written to {output_filename}')

