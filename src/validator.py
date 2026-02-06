import pandas as pd
from datetime import datetime

def validate_extraction(data):
    """
    Runs automated checks on the extracted schedule data.
    Returns a markdown validation report string and a boolean success flag.
    """
    if not data:
        return "## Validation Failed\nNo data to validate.", False

    df = pd.DataFrame(data)
    errors = []
    
    # Check 1: Data Integrity
    # Every row must have course_name, day, and time info
    required_cols = ['course_name', 'day', 'start_time', 'end_time']
    for row_idx, row in df.iterrows():
        missing = [col for col in required_cols if col not in row or pd.isna(row[col]) or str(row[col]).strip() == ""]
        if missing:
            errors.append(f"- Row {row_idx}: Missing required fields: {', '.join(missing)}")

    # Check 2: Logic Check (Start Time < End Time)
    if 'start_time' in df.columns and 'end_time' in df.columns:
        for row_idx, row in df.iterrows():
            start = row['start_time']
            end = row['end_time']
            if start and end:
                try:
                    s_dt = datetime.strptime(start, '%H:%M')
                    e_dt = datetime.strptime(end, '%H:%M')
                    if s_dt >= e_dt:
                        errors.append(f"- Row {row_idx}: Standard Logic Error - Start time ({start}) >= End time ({end}) for {row.get('course_name', 'Unknown')}")
                except ValueError:
                    # Time format error already implied by data integrity or specific format check
                    pass

    # Check 3: Unique Session Check (Room Conflict)
    # Flag instance where same room_id is used by two diff courses at same day/time
    if 'room_id' in df.columns and 'day' in df.columns and 'start_time' in df.columns:
        # Group by Room, Day, StartTime
        # We assume simplified conflict check: exact start time match
        # Ideally we check time range overlap, but prompt asks for "used by two different courses at same day and time"
        dupes = df[df.duplicated(subset=['room_id', 'day', 'start_time'], keep=False)]
        if not dupes.empty:
            grouped = dupes.groupby(['room_id', 'day', 'start_time'])
            for (room, day, time), group in grouped:
                # Ignore generic rooms like DARING, ONLINE, TUBEL (Tugas Belajar), RPS
                room_upper = str(room).upper()
                ignored_rooms = ['DARING', 'ONLINE', 'ZOOM', 'MEET', 'RPS', 'TUBEL', 'MAGANG']
                if any(ignored in room_upper for ignored in ignored_rooms):
                    continue

                courses = group['course_name'].unique()
                if len(courses) > 1 and room: # Only if room is not null
                     errors.append(f"- Room Conflict: Room {room} on {day} at {time} booked for: {', '.join(courses)}")

    # Check 4: Credit Validation
    if 'sks' in df.columns:
        for row_idx, row in df.iterrows():
            try:
                sks = int(row['sks'])
                if not (1 <= sks <= 6):
                    errors.append(f"- Row {row_idx}: SKS Logic Error - SKS {sks} is out of realistic range (1-6)")
            except (ValueError, TypeError):
                 # Covered by type casting in extractor, but failsafe here
                 pass

    # Generate Report
    report = ["# Validation Report"]
    if not errors:
        report.append("✅ **PASSED**: All structural and logical checks passed.")
        success = True
    else:
        report.append(f"❌ **FAILED**: Found {len(errors)} issues.")
        report.append("\n### Issues List:")
        report.extend(errors)
        success = False # Soft fail, we might still want the data but flagged

    return "\n".join(report), success
