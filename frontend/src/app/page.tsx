import Link from 'next/link';

export default function HomePage() {
  return (
    <div>
      <h1>ML Q&A App</h1>
      <p>Navigate to:</p>
      <ul>
        <li>
          <Link href="/learn">Learn Page (Upload CSV)</Link>
        </li>
        <li>
          <Link href="/ask">Ask Page (Get Prediction)</Link>
        </li>
      </ul>
    </div>
  );
}
