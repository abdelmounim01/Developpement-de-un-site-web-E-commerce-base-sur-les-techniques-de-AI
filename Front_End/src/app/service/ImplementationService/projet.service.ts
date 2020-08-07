import { Injectable } from '@angular/core';
import { HttpClient , HttpHeaders  } from '@angular/common/http'
import { Observable } from 'rxjs'
import { Task } from '../Interfaces/task'
import { Info } from '../Interfaces/info'
import { User } from '../Interfaces/user'
import { Metauser } from '../Interfaces/metauser'
import { Router } from '@angular/router'

@Injectable({
  providedIn: 'root'
})
export class ProjetService {

  constructor(private http: HttpClient,private _router: Router) { }

getTasks(): Observable<Task[]> {
    return this.http.get<Task[]>('http://127.0.0.1:5000/api/all')
 }

getTask(produitId:number): Observable<Task[]> {
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/produit/${produitId}`)
}

getInfoCateg(categorie : string): Observable<Task[]>{
return this.http.get<Task[]>(`http://127.0.0.1:5000/api/${categorie}`)
}

getTasksuser(id: number): Observable<Task[]> {
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/all/${id}`)
}

getUsers(): Observable<User[]> {
  return this.http.get<User[]>('http://127.0.0.1:5000/api/users')
}

getMetausers(): Observable<Metauser[]> {
  return this.http.get<Metauser[]>('http://127.0.0.1:5000/api/metausers')
}

getMetauser(userId:Metauser): Observable<Metauser[]> {
  return this.http.get<Metauser[]>(`http://127.0.0.1:5000/api/metauser/${userId}`)
}

addTask(task: Task): Observable<Task> {
    return this.http.post<Task>('http://127.0.0.1:5000/api/task', task)
}

deleteTask(id: number): Observable<{}> {
    const url = `http://127.0.0.1:5000/api/task/${id}`
    return this.http.delete(url)
}

deleteUser(id: number): Observable<{}> {
  const url = `http://127.0.0.1:5000/api/users/${id}`
  return this.http.delete(url)
}

updateTask(task: Task): Observable<Task> {
    const url = `http://127.0.0.1:5000/api/taskupdate/${task.produitId}`
    return this.http.put<Task>(url, task)
}

updateuser(user: User): Observable<User> {
  const url = `http://127.0.0.1:5000/api/updateuser/${user.userId}`
  return this.http.put<User>(url, user)
}

info(): Observable<Info[]>  {
  return this.http.get<Info[]>('http://127.0.0.1:5000/api/info')
}

infouser(id: number): Observable<Info[]>  {
  return this.http.get<Info[]>(`http://127.0.0.1:5000/api/info/${id}`)
}

activeusers(): Observable<User[]>  {
  return this.http.get<User[]>('http://127.0.0.1:5000/api/activeusers')
}

leastactiveusers(): Observable<User[]>  {
  return this.http.get<User[]>('http://127.0.0.1:5000/api/leastactiveusers')
}

popularproducts(): Observable<Task[]>  { /* getMostPopular() */ 
  return this.http.get<Task[]>('http://127.0.0.1:5000/api/tasks')
}

popularproductscat(categorie : string): Observable<Task[]>  { 
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/taskscat/${categorie}`)
}

search(text : string): Observable<Task[]>  { 
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/search/${text}`)
}

popularproductssubcat(categorie : string , subCategorie : string): Observable<Task[]>  {  
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/taskssubcat/${categorie}/${subCategorie}`)
}

popularproductsuser(id: number): Observable<Task[]>  {
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/tasks/${id}`)
}

leastpopularproducts(): Observable<Task[]>  {
  return this.http.get<Task[]>('http://127.0.0.1:5000/api/leasttasks')
}

leastpopularproductsuser(id: User): Observable<Task[]>  {
  return this.http.get<Task[]>(`http://127.0.0.1:5000/api/leasttasks/${id}`)
}

addUser(user) : Observable<User>{
  return this.http.post<User>('http://127.0.0.1:5000/api/users', user)
}

getUser(id: number) : Observable<User>{
  return this.http.get<User>(`http://127.0.0.1:5000/api/user/${id}`)
}

parseTable(files) {
  const formData: FormData = new FormData();
  formData.append('file', files[0], files[0].name);
  return this.http.post('http://127.0.0.1:5000/api/upload_file', formData);
}

addpanier(meta : Metauser): Observable<Metauser> {
  return this.http.post<Metauser>('http://127.0.0.1:5000/api/metausers/panier', meta)
  }
  deletepanier(produitId:number,userId:number): Observable<Metauser> {
    return this.http.delete<Metauser>(`http://127.0.0.1:5000/api/metausers/panierdelete/${produitId}/${userId}`)
    }
  
  addlike(meta : Metauser): Observable<Metauser> {
  return this.http.post<Metauser>('http://127.0.0.1:5000/api/metausers/like', meta)
  }
  
  addrating(meta : Metauser): Observable<Metauser> {
  return this.http.post<Metauser>('http://127.0.0.1:5000/api/metausers/rating', meta)
  }

  getInfoCategie(): Observable<Task[]>{
  return this.http.get<Task[]>('http://127.0.0.1:5000/api/all')
  }
  
  getSubCategory(categorie : string , subCategorie : string): Observable<Task[]>{
    return this.http.get<Task[]>(`http://127.0.0.1:5000/api/${categorie}/${subCategorie}`)
  }

  maylike(iduser:number): Observable<Task[]>{
    return this.http.get<Task[]>(`http://127.0.0.1:5000/api/maylike/${iduser}`)
  }
  
  maylikecat(iduser:number , categorie : string ): Observable<Task[]>{
    return this.http.get<Task[]>(`http://127.0.0.1:5000/api/maylike/${iduser}/${categorie}`)
  }

  maylikesubcat(iduser:number , categorie : string , subCategorie : string): Observable<Task[]>{
    return this.http.get<Task[]>(`http://127.0.0.1:5000/api/maylike/${iduser}/${categorie}/${subCategorie}`)
  }

  similiarproducts(idproducts:number):Observable<Task[]>{
    return this.http.get<Task[]>(`http://127.0.0.1:5000/api/similiar/${idproducts}`)
  }

  messages(message:string):Observable<string>{
    return this.http.get<string>(`http://127.0.0.1:5000/api/chatbot/${message}`)
  }

  images(img:string):Observable<string>{
    return this.http.get<string>(`http://127.0.0.1:5000/api/image/${img}`)
  }

}
