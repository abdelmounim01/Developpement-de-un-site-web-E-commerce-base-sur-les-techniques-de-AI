import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../service/ImplementationService/projet.service'
@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent implements OnInit {
  reponse:any
  rep:any
  collapsed = true;
  constructor(private projetService: ProjetService) { }

  ngOnInit(): void {
    
  }
  message (message: string): void {
    this.projetService.messages(message).subscribe(reponse => (this.reponse = reponse))
    
  }

  
}
